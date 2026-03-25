# Standard library
import heapq
import itertools
import os
import re
import threading
import time
import uuid
import json
import hashlib
import logging
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Optional, Tuple, Any
from queue import Queue, Full, Empty

# Third-party
import numpy as np
import torch
from tqdm import tqdm

# Local / project
from openai import OpenAI
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.utils.metric import reduce_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.rollout_skip import RolloutSkip

try:
    from tensordict import TensorDict  # preferred
except Exception:
    from torchrl.data import TensorDict  # fallback for older installs


def _to_td(batch_dict: Dict[str, torch.Tensor]) -> TensorDict:
    """Wrap a plain dict of tensors into a TensorDict with the right batch_size."""
    # infer B from the first tensor we find
    B = None
    for v in batch_dict.values():
        if isinstance(v, torch.Tensor):
            B = v.size(0)
            break
    if B is None:
        B = 0
    return TensorDict(batch_dict, batch_size=[B])


def _to_indexable_array(v, n):
    """
    Convert v into something indexable by DataProto.__getitem__:
    - lists/tuples/ndarrays => np.array with correct length (pad/trim)
    - scalars => broadcast to length n
    - dict => broadcast an object array [dict]*n
    """
    import numpy as _np

    # Already a numpy array of correct length
    if isinstance(v, _np.ndarray):
        if v.ndim == 0:
            return _np.repeat(v.reshape(1), n)
        if len(v) == n:
            if v.ndim == 1:
                return v
            # Convert each row to an object slot to keep it 1-D
            out = _np.empty(n, dtype=object)
            for i in range(n):
                out[i] = v[i]
            return out
        if len(v) == 1:
            return _np.repeat(v, n, axis=0)
        # fallback pad/trim to 1-D object array
        out = _np.empty(n, dtype=object)
        m = len(v)
        for i in range(n):
            out[i] = v[i] if i < m else v[-1]
        return out

    # Python list/tuple
    if isinstance(v, (list, tuple)):
        if len(v) == n:
            return _np.array(v, dtype=object)
        if len(v) == 1:
            return _np.array(list(v) * n, dtype=object)
        out = _np.empty(n, dtype=object)
        m = len(v)
        for i in range(n):
            out[i] = v[i] if i < m else v[-1]
        return out

    # A dict (per-item rich object) → broadcast it
    if isinstance(v, dict):
        return _np.array([v] * n, dtype=object)

    # Scalar/anything else → broadcast
    return _np.array([v] * n, dtype=object)


def _sanitize_non_tensor_batch(dp):
    try:
        n = len(dp.batch["input_ids"])
    except Exception:
        # If we can't read length, do nothing
        return dp
    for k, v in list(dp.non_tensor_batch.items()):
        # Ensure every value becomes indexable and aligned to n
        dp.non_tensor_batch[k] = _to_indexable_array(v, n)
    return dp


def _extract_text_from_prompt_field(prompt_field) -> str:
    """Extract a plain problem string from the dataset's 'prompt' field."""
    try:
        # Case: list of chat messages [{"role": "...", "content": "..."}]
        if isinstance(prompt_field, list):
            # Prefer the last user message; otherwise first content
            for msg in reversed(prompt_field):
                if isinstance(msg, dict) and msg.get("role") == "user" and "content" in msg:
                    return str(msg["content"])
            if prompt_field and isinstance(prompt_field[0], dict) and "content" in prompt_field[0]:
                return str(prompt_field[0]["content"])
            # List of strings fallback
            return " ".join(map(str, prompt_field))
        # Case: direct string
        if isinstance(prompt_field, str):
            return prompt_field
    except Exception:
        pass
    return ""


def _normalize_reward_model_list(rm, n, extra_info_list=None):
    """
    Ensure we have a length-n object array of dicts like:
      [{"ground_truth": str|None, "style": str|None}, ...]
    If rm is a single dict (dataset format), broadcast it. If missing GT, try extra_info.raw_answer.
    """
    def _gt_from_extra(extra):
        if isinstance(extra, dict):
            return extra.get("raw_answer")
        return None

    # Convert to list of dicts
    if isinstance(rm, dict):
        rm_list = [rm] * n
    elif isinstance(rm, (list, tuple, np.ndarray)):
        rm_list = list(rm)
        if len(rm_list) == 0:
            rm_list = [{}] * n
        elif len(rm_list) != n:
            # pad/trim
            if len(rm_list) < n:
                rm_list = rm_list + [rm_list[-1]] * (n - len(rm_list))
            else:
                rm_list = rm_list[:n]
    else:
        rm_list = [{}] * n

    out = []
    for i in range(n):
        d = dict(rm_list[i] if isinstance(rm_list[i], dict) else {})
        # fill ground_truth if missing using extra_info.raw_answer
        if "ground_truth" not in d or d["ground_truth"] in (None, ""):
            if extra_info_list and i < len(extra_info_list):
                gt = _gt_from_extra(extra_info_list[i])
                if gt is not None:
                    d["ground_truth"] = gt
        # always keep style key present for downstream consumers
        if "style" not in d:
            d["style"] = None
        out.append(d)
    return np.asarray(out, dtype=object)


def _build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Standard GPT-style position_ids for left-padding masks, robust to empty/1D inputs.
    pos = cumsum(attention_mask) - 1, zeros where mask==0, clamped >= 0.
    """
    if attention_mask.numel() == 0:
        # Preserve shape; make sure it's long dtype
        return torch.zeros_like(attention_mask, dtype=torch.long)
    if attention_mask.dim() == 1:
        # Promote to (B=1, L) so cumsum(dim=1) is valid
        am = attention_mask.unsqueeze(0).long()
        pos = am.cumsum(dim=1) - 1
        pos = torch.where(am > 0, pos, torch.zeros_like(pos))
        pos = pos.clamp_min_(0)
        return pos.squeeze(0)
    # Usual 2D case
    pos = attention_mask.long().cumsum(dim=1) - 1
    pos = torch.where(attention_mask > 0, pos, torch.zeros_like(pos))
    return pos.clamp_min_(0)


# Configure logging
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================
# Data dumping utilities
# ==============================

class AugmentationLogger:
    """Logs augmentation data for offline analysis."""

    def __init__(self, log_dir: str, experiment_name: str, max_buffer_size: int = 500):
        self.log_dir = Path(log_dir) / "augmentation_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # File paths
        self.augmentation_file = self.log_dir / \
            f"{experiment_name}_{self.timestamp}_augmentations.jsonl"
        self.annotation_file = self.log_dir / \
            f"{experiment_name}_{self.timestamp}_annotations.jsonl"
        self.pool_snapshot_file = self.log_dir / \
            f"{experiment_name}_{self.timestamp}_pool_snapshots.jsonl"

        # Buffers for batch writing
        self.augmentation_buffer = []
        self.annotation_buffer = []
        self.pool_snapshot_buffer = []
        self.max_buffer_size = max_buffer_size
        self._lock = threading.Lock()  # Add thread safety

        # Statistics
        self.stats = {
            "total_augmentations": 0,
            "total_annotations": 0,
            "total_solvable": 0,
            "total_unsolvable": 0,
            "avg_estimated_reward": 0.0,
            "avg_final_reward": 0.0,
            "reward_correlation": [],
            "pool_snapshots": 0,
        }

        # Create metadata file
        metadata = {
            "experiment_name": experiment_name,
            "timestamp": self.timestamp,
            "log_dir": str(self.log_dir),
            "files": {
                "augmentations": str(self.augmentation_file),
                "annotations": str(self.annotation_file),
                "pool_snapshots": str(self.pool_snapshot_file)
            }
        }

        with open(self.log_dir / f"{experiment_name}_{self.timestamp}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"AugmentationLogger initialized at {self.log_dir}")

    def log_augmentation(self, record: Dict[str, Any]):
        """Log an augmentation event."""
        with self._lock:
            record["timestamp"] = datetime.now().isoformat()
            record["id"] = str(uuid.uuid4())

            self.augmentation_buffer.append(record)
            self.stats["total_augmentations"] += 1

            if "estimated_reward" in record:
                # Update running average
                c = self.stats.get("est_reward_count", 0)
                prev = self.stats.get("avg_estimated_reward", 0.0)
                try:
                    val = float(record["estimated_reward"])
                except Exception:
                    val = None
                if val is not None and np.isfinite(val):
                    self.stats["avg_estimated_reward"] = (
                        prev * c + val) / (c + 1)
                    self.stats["est_reward_count"] = c + 1

            if len(self.augmentation_buffer) >= self.max_buffer_size:
                self._flush_augmentation_buffer()

    def log_annotation(self, record: Dict[str, Any]):
        """Log a teacher annotation event."""
        with self._lock:
            record["timestamp"] = datetime.now().isoformat()
            record["id"] = str(uuid.uuid4())

            self.annotation_buffer.append(record)
            self.stats["total_annotations"] += 1

            if record.get("solvable", False):
                self.stats["total_solvable"] += 1
            else:
                self.stats["total_unsolvable"] += 1

            # Track reward correlation
            if "estimated_reward" in record and "final_reward" in record:
                self.stats["reward_correlation"].append({
                    "estimated": record["estimated_reward"],
                    "final": record["final_reward"]
                })

                # Update running average of final rewards
                n = len(self.stats["reward_correlation"])
                if n > 0:
                    self.stats["avg_final_reward"] = sum(
                        r["final"] for r in self.stats["reward_correlation"]
                    ) / n

            if len(self.annotation_buffer) >= self.max_buffer_size:
                self._flush_annotation_buffer()

    def log_pool_snapshot(self, pool_metrics: Dict[str, Any], sample: List[Dict[str, Any]] = None):
        """Log a snapshot of the query pool state."""
        with self._lock:
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "metrics": pool_metrics,
                # Log up to 10 samples
                "sample": sample[:10] if sample else [],
            }

            self.pool_snapshot_buffer.append(snapshot)
            self.stats["pool_snapshots"] += 1

            if len(self.pool_snapshot_buffer) >= self.max_buffer_size // 10:  # Less frequent
                self._flush_pool_snapshot_buffer()

    def _flush_augmentation_buffer(self):
        """Write augmentation buffer to file."""
        if not self.augmentation_buffer:
            return

        try:
            with open(self.augmentation_file, 'a') as f:
                for record in self.augmentation_buffer:
                    f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Failed to flush augmentation buffer: {e}")

        self.augmentation_buffer.clear()

    def _flush_annotation_buffer(self):
        """Write annotation buffer to file."""
        if not self.annotation_buffer:
            return

        try:
            with open(self.annotation_file, 'a') as f:
                for record in self.annotation_buffer:
                    f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Failed to flush annotation buffer: {e}")

        self.annotation_buffer.clear()

    def _flush_pool_snapshot_buffer(self):
        """Write pool snapshot buffer to file."""
        if not self.pool_snapshot_buffer:
            return

        try:
            with open(self.pool_snapshot_file, 'a') as f:
                for record in self.pool_snapshot_buffer:
                    f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Failed to flush pool snapshot buffer: {e}")

        self.pool_snapshot_buffer.clear()

    def flush_all(self):
        """Flush all buffers to disk."""
        with self._lock:
            self._flush_augmentation_buffer()
            self._flush_annotation_buffer()
            self._flush_pool_snapshot_buffer()


# ==============================
# Enhanced Query Record with tracking
# ==============================

@dataclass
class QueryRecord:
    """Lightweight record kept on the driver for queue items."""
    raw_prompt_data: np.ndarray          # Can be token ids or text string
    # cached tensors for fast batching (text-only)
    input_ids: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    # teacher-provided ground truth (task-dependent)
    gt: Optional[object] = None
    # NOTE: no unreliable bootstrapping; we learn this post-rollout
    reward: Optional[float] = None
    # policy-estimated reward for augmented queries (will mirror reward when set)
    est_reward: Optional[float] = None
    meta: dict = field(default_factory=dict)

    # New fields for tracking
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_text: Optional[str] = None
    augmented_text: Optional[str] = None
    teacher_response: Optional[str] = None
    creation_time: str = field(
        default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "record_id": self.record_id,
            "reward": self.reward,
            "est_reward": self.est_reward,
            "gt": str(self.gt) if self.gt is not None else None,
            "original_text": self.original_text,
            "augmented_text": self.augmented_text,
            "teacher_response": self.teacher_response,
            "creation_time": self.creation_time,
            "meta": self.meta,
        }


# ==============================
# Heap-based query pool
# ==============================


class ThreadSafeQueryPool:
    """
    Two partitions with reward-aware sampling.

    low_heap  : key = -reward (acts like max-heap on reward). Top is the *largest* reward
                within the low side → "close-bottom" (medium-ish) of low partition.
    high_heap : true min-heap **and** mirror max-heap with lazy deletion for O(log n) max-evictions.

    Full-insert rule:
      - If pool full and new.reward > global_max_reward: ignore (too easy).
      - Else insert into low/high depending on comparisons with the two fronts,
        then evict the *globally highest* reward to keep size constant.
    """

    def __init__(
        self,
        max_size: int = 30000,
        low_fraction: float = 0.5,
        rng: Optional[np.random.Generator] = None,
        cleanup_frequency: int = 1000,  # Clean up stale entries every N operations
        mixed_easy_medium: bool = False
    ):
        self._lock = threading.RLock()  # Use RLock to prevent deadlocks
        self._max_size = max(1, int(max_size))  # Ensure at least 1
        self._low_fraction = float(np.clip(low_fraction, 0.05, 0.95))
        self._cleanup_frequency = cleanup_frequency
        self._operations_count = 0
        self._mixed_easy_medium = bool(mixed_easy_medium)

        # Heaps hold tuples: (key, seq, QueryRecord)
        self._low_heap: List[Tuple[float, int, QueryRecord]] = []

        # High side uses twin heaps + lazy deletion for efficient max removal
        self._high_heap_min: List[Tuple[float, int, QueryRecord]] = []
        self._high_heap_max: List[Tuple[float, int, QueryRecord]] = []
        self._active_high: set[int] = set()

        # “cold” items with reward=None (FIFO to get them scored quickly)
        self._cold_queue: deque[QueryRecord] = deque()

        self._seq = itertools.count()
        self._rng = rng if rng is not None else np.random.default_rng()

        # Metrics tracking
        self._total_added = 0
        self._total_sampled = 0
        self._total_evicted = 0

    def set_mixed_easy_medium(self, enabled: bool) -> None:
        with self._lock:
            self._mixed_easy_medium = bool(enabled)

    # ---------- capacity / admin ----------

    def set_max_size(self, max_size: int):
        with self._lock:
            old_size = self._max_size
            self._max_size = max(1, int(max_size))  # Ensure at least 1
            if self._max_size < old_size:
                self._evict_to_capacity_unlocked()
            logger.info(
                f"Pool max size changed from {old_size} to {self._max_size}")

    def _high_size_unlocked(self) -> int:
        return len(self._active_high)

    def capacity_remaining(self) -> int:
        with self._lock:
            return max(0, self._max_size - (len(self._low_heap) + self._high_size_unlocked() + len(self._cold_queue)))

    def size(self) -> int:
        with self._lock:
            return len(self._low_heap) + self._high_size_unlocked() + len(self._cold_queue)

    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_size": self.size(),
                "low_heap_size": len(self._low_heap),
                "high_heap_size": self._high_size_unlocked(),
                "cold_queue_size": len(self._cold_queue),  # NEW
                "capacity_remaining": self.capacity_remaining(),
                "total_added": self._total_added,
                "total_sampled": self._total_sampled,
                "total_evicted": self._total_evicted,
            }

    # ---------- initialization ----------

    def initialize_uniform(self, seed_items: List[QueryRecord]):
        # drop uniformization; just add as-is (many will have reward=None)
        self.add_many(seed_items)
        logger.info(
            f"Initialized pool with {len(seed_items)} items (no uniform rewards)")

    # ---------- public mutation ----------

    def add_many(self, items: List[QueryRecord]):
        if not items:
            return
        added_count = 0
        with self._lock:
            for it in items:
                total = len(self._low_heap) + \
                    self._high_size_unlocked() + len(self._cold_queue)
                if total < self._max_size:
                    if it.reward is None:
                        self._cold_queue.append(it)
                    else:
                        r = float(it.reward)
                        front = self._peek_high_front_reward_unlocked()
                        if front is None or r > front:
                            self._push_high_unlocked(it)
                        else:
                            self._push_low_unlocked(it)
                    self._rebalance_low_high_unlocked(local_only=True)
                    added_count += 1
                else:
                    # pool full
                    if it.reward is None:
                        # cannot order; drop oldest “too easy” scored item first, else pop oldest cold
                        if not self._remove_high_heap_max_unlocked():
                            if self._low_heap:
                                heapq.heappop(self._low_heap)
                                self._total_evicted += 1
                            elif self._cold_queue:
                                self._cold_queue.popleft()
                                self._total_evicted += 1
                        self._cold_queue.append(it)
                        added_count += 1
                    else:
                        if self._insert_when_full_unlocked(it):
                            added_count += 1

            self._total_added += added_count
            self._operations_count += len(items)
            if self._operations_count >= self._cleanup_frequency:
                self._deep_clean_heaps_unlocked()
                self._operations_count = 0
            self._evict_to_capacity_unlocked()

        if added_count < len(items):
            logger.debug(
                f"Added {added_count}/{len(items)} items (pool at capacity)")

    # ---------- public sampling ----------

    def _sample_medium_k_unlocked(self, k: int) -> List[QueryRecord]:
        if k <= 0:
            return []

        n_total = len(self._low_heap) + self._high_size_unlocked()
        if n_total == 0:
            return []

        actual_k = min(k, n_total)

        if actual_k >= n_total:
            candidates, origins = self._pop_candidates_unlocked(n_total)
            chosen_idx = list(range(len(candidates)))
        else:
            take = min(2 * actual_k, n_total)
            candidates, origins = self._pop_candidates_unlocked(take)
            if len(candidates) > 0:
                chosen_idx = self._rng.choice(
                    len(candidates),
                    size=min(actual_k, len(candidates)),
                    replace=False
                ).tolist()
            else:
                chosen_idx = []

        chosen = [candidates[i] for i in chosen_idx]

        keep_mask = np.ones(len(candidates), dtype=bool)
        for idx in chosen_idx:
            if idx < len(keep_mask):
                keep_mask[idx] = False
        to_reinsert = [c for i, c in enumerate(
            candidates) if i < len(keep_mask) and keep_mask[i]]
        origins_back = [origins[i] for i in range(
            len(origins)) if i < len(keep_mask) and keep_mask[i]]

        # Reinsert non-chosen items
        for rec, origin in zip(to_reinsert, origins_back):
            if origin == "low":
                self._push_low_unlocked(rec)
            else:
                self._push_high_unlocked(rec)

        self._rebalance_low_high_unlocked(local_only=True)
        return chosen

    def sample_batch(self, k: int) -> List[QueryRecord]:
        if k <= 0:
            return []
        with self._lock:
            n_total = len(self._low_heap) + \
                self._high_size_unlocked() + len(self._cold_queue)
            if n_total == 0:
                return []

            actual_k = min(k, n_total)

            # 1) Pull from cold queue first (get them scored)
            cold_take = min(actual_k, len(self._cold_queue))
            cold = [self._cold_queue.popleft() for _ in range(cold_take)]

            # 2) Fill remainder via existing medium/mixed policy
            need = actual_k - cold_take
            if need <= 0:
                chosen = cold
            else:
                if getattr(self, "_mixed_easy_medium", False):
                    want_easy = need // 2
                    easy: List[QueryRecord] = []
                    for _ in range(want_easy):
                        popped = self._pop_high_max_unlocked()
                        if popped is None:
                            break
                        _, _, it = popped
                        easy.append(it)
                    medium = self._sample_medium_k_unlocked(need - len(easy))
                    chosen = cold + easy + medium
                    self._rebalance_low_high_unlocked(local_only=True)
                else:
                    medium = self._sample_medium_k_unlocked(need)
                    chosen = cold + medium

            self._total_sampled += len(chosen)
            return [self._copy_record_unlocked(r) for r in chosen]

    # ---------- internal helpers (heap ops) ----------

    def _pop_high_max_unlocked(self):
        self._clean_high_max_top_unlocked()
        while self._high_heap_max:
            neg_r, seq, it = heapq.heappop(
                self._high_heap_max)  # neg_r = -reward
            if seq in self._active_high:
                self._active_high.remove(seq)
                return (-float(neg_r), seq, it)
        return None

    def _push_low_unlocked(self, it: QueryRecord):
        heapq.heappush(
            self._low_heap, (-float(it.reward), next(self._seq), it))

    def _push_high_unlocked(self, it: QueryRecord):
        seq = next(self._seq)
        reward = float(it.reward)
        heapq.heappush(self._high_heap_min, (reward, seq, it))
        heapq.heappush(self._high_heap_max, (-reward, seq, it))
        self._active_high.add(seq)

    def _clean_high_min_top_unlocked(self):
        while self._high_heap_min and (self._high_heap_min[0][1] not in self._active_high):
            heapq.heappop(self._high_heap_min)

    def _clean_high_max_top_unlocked(self):
        while self._high_heap_max and (self._high_heap_max[0][1] not in self._active_high):
            heapq.heappop(self._high_heap_max)

    def _deep_clean_heaps_unlocked(self):
        """Periodically rebuild heaps to remove all stale entries (memory efficient)."""
        # Clean in batches to avoid memory spike
        batch_size = 10000

        # Clean high_heap_min
        if len(self._high_heap_min) > batch_size or len(self._high_heap_min) > 2*len(self._active_high):
            valid_min = []
            for r, s, it in self._high_heap_min:
                if s in self._active_high:
                    valid_min.append((r, s, it))
            self._high_heap_min = valid_min
            heapq.heapify(self._high_heap_min)

        # Clean high_heap_max
        if len(self._high_heap_max) > batch_size or len(self._high_heap_max) > 2*len(self._active_high):
            valid_max = []
            for r, s, it in self._high_heap_max:
                if s in self._active_high:
                    valid_max.append((r, s, it))
            self._high_heap_max = valid_max
            heapq.heapify(self._high_heap_max)

    def _pop_high_min_unlocked(self) -> Optional[Tuple[float, int, QueryRecord]]:
        self._clean_high_min_top_unlocked()
        if not self._high_heap_min:
            return None
        reward, seq, it = heapq.heappop(self._high_heap_min)
        if seq in self._active_high:
            self._active_high.remove(seq)
            return (reward, seq, it)
        return self._pop_high_min_unlocked()

    def _pop_candidates_unlocked(self, take: int):
        candidates: List[QueryRecord] = []
        origins: List[str] = []
        want_high = True

        while take > 0 and (self._low_heap or self._active_high):
            pulled = False

            if want_high and self._active_high:
                popped = self._pop_high_min_unlocked()
                if popped is not None:
                    _, _, it = popped
                    candidates.append(it)
                    origins.append("high")
                    take -= 1
                    pulled = True
            elif (not want_high) and self._low_heap:
                _, _, it = heapq.heappop(self._low_heap)
                candidates.append(it)
                origins.append("low")
                take -= 1
                pulled = True

            if not pulled:
                # Fallback to whatever is available
                if self._active_high:
                    popped = self._pop_high_min_unlocked()
                    if popped is not None:
                        _, _, it = popped
                        candidates.append(it)
                        origins.append("high")
                        take -= 1
                elif self._low_heap:
                    _, _, it = heapq.heappop(self._low_heap)
                    candidates.append(it)
                    origins.append("low")
                    take -= 1
                else:
                    break

            want_high = not want_high

        return candidates, origins

    def _peek_low_front_reward_unlocked(self) -> Optional[float]:
        if not self._low_heap:
            return None
        return -self._low_heap[0][0]

    def _peek_high_front_reward_unlocked(self) -> Optional[float]:
        self._clean_high_min_top_unlocked()
        if not self._high_heap_min:
            return None
        return self._high_heap_min[0][0]

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
            _, seq, _ = heapq.heappop(self._high_heap_max)
            if seq in self._active_high:
                self._active_high.remove(seq)
                self._total_evicted += 1
                return True
        return False

    def _insert_when_full_unlocked(self, it: QueryRecord) -> bool:
        """Insert item when pool is full. Returns True if inserted."""
        gmax = self._global_max_reward_unlocked()
        if gmax is not None and float(it.reward) > gmax:
            return False  # Reject too easy items

        r_low = self._peek_low_front_reward_unlocked()
        r_high = self._peek_high_front_reward_unlocked()

        # Determine where to insert
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

        # Evict highest reward item
        if not self._remove_high_heap_max_unlocked():
            if self._low_heap:
                heapq.heappop(self._low_heap)
                self._total_evicted += 1

        self._rebalance_low_high_unlocked(local_only=True)
        return True

    def _rebalance_low_high_unlocked(self, local_only: bool = False):
        """Rebalance heaps to maintain target ratio (batch operations for efficiency)."""
        total = len(self._low_heap) + self._high_size_unlocked()
        if total == 0:
            return

        target_low = int(round(self._low_fraction * total))
        current_low = len(self._low_heap)

        if abs(current_low - target_low) <= 1:  # Close enough
            return

        # Batch transfers for efficiency
        if current_low < target_low:
            # Move from high to low
            to_move = min(target_low - current_low, self._high_size_unlocked())
            if local_only:
                to_move = min(to_move, 4)
            for _ in range(to_move):
                popped = self._pop_high_min_unlocked()
                if popped is None:
                    break
                _, _, it = popped
                self._push_low_unlocked(it)
        elif current_low > target_low:
            # Move from low to high
            to_move = min(current_low - target_low, len(self._low_heap))
            if local_only:
                to_move = min(to_move, 4)
            for _ in range(to_move):
                if not self._low_heap:
                    break
                _, _, it = heapq.heappop(self._low_heap)
                self._push_high_unlocked(it)

    def _evict_to_capacity_unlocked(self):
        total = len(self._low_heap) + \
            self._high_size_unlocked() + len(self._cold_queue)
        if total <= self._max_size:
            return

        need = total - self._max_size
        evicted = 0

        while need > 0 and self._active_high:
            if self._remove_high_heap_max_unlocked():
                evicted += 1
                need -= 1

        while need > 0 and self._low_heap:
            heapq.heappop(self._low_heap)
            self._total_evicted += 1
            evicted += 1
            need -= 1

        while need > 0 and self._cold_queue:
            self._cold_queue.popleft()
            self._total_evicted += 1
            evicted += 1
            need -= 1

        if evicted > 0:
            logger.debug(f"Evicted {evicted} items to maintain capacity")

    @staticmethod
    def _copy_record_unlocked(it: QueryRecord) -> QueryRecord:
        return QueryRecord(
            raw_prompt_data=it.raw_prompt_data.copy() if isinstance(
                it.raw_prompt_data, np.ndarray) else it.raw_prompt_data,
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
        """Get a sample of pool items for logging."""
        with self._lock:
            samples = []

            # Sample from low heap
            for i in range(min(n//2, len(self._low_heap))):
                _, _, record = self._low_heap[i]
                samples.append(record.to_dict())

            # Sample from high heap
            high_samples = []
            for reward, seq, record in self._high_heap_min:
                if seq in self._active_high:
                    high_samples.append(record.to_dict())
                    if len(high_samples) >= n//2:
                        break

            samples.extend(high_samples)
            return samples

# ==============================
# Asynchronous GPT annotator with retry and backpressure
# ==============================


class AsyncTeacherAnnotator(threading.Thread):
    """
    Background thread with improvements:
      - Bounded queue with backpressure
      - Retry logic for API calls
      - Proper error handling and logging
      - Graceful shutdown
      - Data logging for offline analysis
    """

    def __init__(
        self,
        trainer_ref: "RayDAPOTrainer",
        poll_interval: float = 0.1,
        max_queue: int = 20000,
        api_timeout: float = 30.0,
        model_name: str = "gpt-5-mini",
        max_retries: int = 1,
        retry_delay: float = 1.0,
        augmentation_logger: Optional[AugmentationLogger] = None
    ):
        super().__init__(daemon=True)
        self.trainer_ref = trainer_ref
        self.queue = Queue(maxsize=max_queue)
        self.stop_event = threading.Event()
        self.poll_interval = poll_interval
        self.api_timeout = api_timeout
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._running = True
        self._shutdown_lock = threading.Lock()
        self.augmentation_logger = augmentation_logger

        # Metrics
        self._processed_count = 0
        self._error_count = 0
        self._api_call_count = 0

        # OpenAI client
        api_key = os.environ.get("OPENAI_API_KEY", "")

        self.client = OpenAI(api_key=api_key, timeout=api_timeout)

        # Thread pool for parallel API calls
        self.executor = ThreadPoolExecutor(max_workers=8)

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """
        Return True if `text` contains Chinese/CJK characters or common full-width punctuation.
        (BMP ranges; good enough for filtering training data.)
        """
        if not text:
            return False
        return bool(re.search(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]', text))

    def enqueue_aug(self, aug_proto: "DataProto") -> bool:
        """Enqueue with backpressure. Returns False if queue is full."""
        try:
            self.queue.put(aug_proto, block=False)
            return True
        except Full:
            logger.warning(
                "Teacher annotation queue is full, applying backpressure")
            return False

    def shutdown(self):
        """Graceful shutdown with timeout."""
        with self._shutdown_lock:
            if not self._running:
                return

            self._running = False
            self.stop_event.set()

            # Process remaining items with timeout
            timeout = 5.0
            start = time.time()
            while not self.queue.empty() and (time.time() - start) < timeout:
                time.sleep(0.1)

            # Shutdown executor
            self.executor.shutdown(wait=True)

            if not self.queue.empty():
                logger.warning(
                    f"Shutting down with {self.queue.qsize()} items still in queue")

    def get_metrics(self) -> Dict[str, Any]:
        """Get annotator metrics."""
        return {
            "queue_size": self.queue.qsize(),
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "api_call_count": self._api_call_count,
        }

    def _make_extract_and_solve_prompt(self, original: str, generation: str) -> str:
        return (
            "You are a math data cleaner and solver.\n"
            "TASKS:\n"
            "1) Read ORIGINAL and GENERATION. Extract a single, self-contained math problem statement from GENERATION only. "
            "Remove any prefaces, commentary (e.g., 'Question:', 'Assistant:'), code fences, and any 'Answer:' lines. "
            "Don't copy any text from ORIGINAL. If no clean question can be extracted, return an empty string and mark unsolvable.\n"
            "2) If a clean question exists, decide if it is well-posed. If solvable, compute ONLY the final numeric answer.\n"
            "3) Estimate relative difficulty vs ORIGINAL on a 0.75-1.33 scale (1.0=same).\n\n"
            "Return ONLY one JSON object on a single line:\n"
            '{"clean":"<string>","solvable":true|false,"answer":"<string or null>","difficulty":<number>}\n'
            "Rules: lowercase true/false/null; 'answer' must be a bare number string when solvable, else null.\n\n"
            f"ORIGINAL:\n{original}\n\nGENERATION:\n{generation}"
        )

    def _extract_text_and_reason(self, response) -> tuple[str, str | None]:
        """
        Return (content, finish_reason). Works across SDK variants.
        - If JSON-mode produced a parsed object, we stringify it.
        """
        try:
            if hasattr(response, "choices") and response.choices:
                ch = response.choices[0]
                # new SDKs may have finish_details.type
                fr = getattr(ch, "finish_reason", None)
                if fr is None:
                    fr = getattr(getattr(ch, "finish_details", {}), "type", None) \
                        or (getattr(ch, "finish_details", {}) or {}).get("type")
                msg = getattr(ch, "message", None)
                if msg is None:
                    return "", fr
                # some SDKs expose structured JSON as message.parsed
                parsed = getattr(msg, "parsed", None)
                if parsed:
                    try:
                        return json.dumps(parsed, ensure_ascii=False), fr
                    except Exception:
                        pass
                # fallback to plain content
                if hasattr(msg, "content"):
                    return (msg.content or "").strip(), fr
                if isinstance(msg, dict):
                    return (msg.get("content", "") or "").strip(), fr
        except Exception as e:
            logger.error(f"Error extracting text/reason: {e}")
        return "", None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```") and t.endswith("```"):
            t = t[3:-3].strip()
        t = re.sub(r"^json\s*", "", t, flags=re.IGNORECASE)
        return t

    @staticmethod
    def _valid_est(x) -> bool:
        try:
            xv = float(x)
            return np.isfinite(xv)
        except (TypeError, ValueError):
            return False

    def _call_extract_solve_with_retry(self, original: str, generation: str):
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
                    max_completion_tokens=4196,
                    response_format={"type": "json_object"},
                    reasoning_effort="low",
                )
                content, finish_reason = self._extract_text_and_reason(resp)
                text = self._strip_code_fences(content).strip()
                obj = json.loads(text)

                clean = (obj.get("clean") or "").strip()
                solvable = bool(obj.get("solvable", False))
                ans = obj.get("answer", None)
                diff = obj.get("difficulty", 1.0)
                try:
                    diff = float(diff)
                except Exception:
                    diff = 1.0
                if not np.isfinite(diff):
                    diff = 1.0
                diff = float(np.clip(diff, 0.75, 1.33))

                return {
                    "ok": True,
                    "clean": clean,
                    "solvable": solvable,
                    "answer": (str(ans) if isinstance(ans, str) else None) if solvable else None,
                    "difficulty": diff,
                    "raw": content,
                    "finish_reason": finish_reason,
                }
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    return {"ok": False, "error": str(e), "raw": ""}

    def run(self):
        """Main annotation loop with proper synchronization."""
        while self._running:
            try:
                # Get item with timeout to allow checking stop_event
                aug_proto = self.queue.get(timeout=self.poll_interval)
            except Empty:
                if self.stop_event.is_set():
                    break
                continue

            if not self._running:
                self.queue.task_done()
                break

            try:
                # Process the augmented proto
                self._process_augmented_proto(aug_proto)
                self._processed_count += 1
            except Exception as e:
                logger.error(
                    f"Error processing augmented proto: {e}", exc_info=True)
                self._error_count += 1
            finally:
                self.queue.task_done()

    def _process_augmented_proto(self, aug_proto: DataProto):
        """Process a single augmented proto with logging."""
        # Select only items that have an estimated reward
        est = aug_proto.non_tensor_batch.get("policy/est_reward", None)
        if est is None:
            return

        est = list(est)
        idxs = [i for i, v in enumerate(est) if self._valid_est(v)]
        if not idxs:
            return

        # Narrow the proto to only those needing annotation
        _sanitize_non_tensor_batch(aug_proto)
        sub_proto = aug_proto[idxs]

        # Get original and augmented texts for logging
        original_texts = list(
            sub_proto.non_tensor_batch.get("original_text", []))
        augmented_texts = list(
            sub_proto.non_tensor_batch.get("raw_prompt_data", []))

        # Ensure lists have the same length as idxs
        original_texts = original_texts[:len(
            idxs)] + [None] * max(0, len(idxs) - len(original_texts))
        augmented_texts = augmented_texts[:len(
            idxs)] + [None] * max(0, len(idxs) - len(augmented_texts))

        # Teacher annotation via OpenAI API (parallel)
        raw_prompts = augmented_texts

        # Preallocate outputs aligned to sub_proto order
        N = len(raw_prompts)
        teacher_answers = [None] * N
        teacher_solvable = [False] * N  # default to unsolvable
        teacher_raw_responses = [""] * N
        clean_questions = ["" for _ in range(N)]
        teacher_difficulty = [1.0 for _ in range(N)]
        futures = []

        for i, q in enumerate(raw_prompts):
            if not self._running:
                break
            q_text = str(q)
            # keep your CJK hard filter
            if self._contains_chinese(q_text):
                teacher_solvable[i] = False
                teacher_answers[i] = None
                teacher_raw_responses[i] = "[filtered:contains_chinese]"
                if self.augmentation_logger and i < len(idxs):
                    self.augmentation_logger.log_annotation({
                        "original_text": original_texts[i] if i < len(original_texts) else None,
                        "augmented_text": q_text,
                        "estimated_reward": float(est[idxs[i]]) if idxs[i] < len(est) else None,
                        "teacher_model": self.model_name,
                        "solvable": False,
                        "teacher_answer": None,
                        "teacher_raw_response": "[filtered:contains_chinese]",
                        "reason": "contains_chinese_characters",
                    })
                continue

            orig = str(original_texts[i]) if i < len(original_texts) else ""
            fut = self.executor.submit(
                self._call_extract_solve_with_retry, orig, q_text)
            futures.append((i, fut, q_text))

        # Collect results from API calls and write them back into the preallocated slots
        for i, future, question_text in futures:
            try:
                r = future.result(timeout=self.api_timeout * 2)
                if not r.get("ok", False):
                    teacher_solvable[i] = False
                    teacher_answers[i] = None
                    teacher_raw_responses[i] = r.get("raw", "")
                    continue

                ans = r.get("answer")
                is_solvable = bool(r.get("solvable", False))
                has_ans = isinstance(ans, str) and ans.strip() != ""

                clean_questions[i] = r["clean"]
                teacher_difficulty[i] = r["difficulty"]
                teacher_raw_responses[i] = r["raw"]

                # Gate obvious bad cases
                if not clean_questions[i]:
                    teacher_solvable[i] = False
                    teacher_answers[i] = None
                    continue
                if is_solvable and not has_ans:
                    # Treat as unsolvable; don't let None reach downstream
                    teacher_solvable[i] = False
                    teacher_answers[i] = None
                    teacher_raw_responses[i] = r.get("raw", "")
                    continue

                teacher_solvable[i] = is_solvable
                teacher_answers[i] = ans if is_solvable else None

                # Log annotation event with richer fields
                if self.augmentation_logger and i < len(idxs):
                    self.augmentation_logger.log_annotation({
                        "original_text": original_texts[i] if i < len(original_texts) else None,
                        "augmented_text": question_text,
                        "cleaned_question": clean_questions[i],
                        "estimated_reward": float(est[idxs[i]]) if idxs[i] < len(est) else None,
                        "teacher_model": self.model_name,
                        "solvable": teacher_solvable[i],
                        "teacher_answer": teacher_answers[i],
                        "teacher_raw_response": r["raw"],
                        "teacher_difficulty": teacher_difficulty[i],
                        "api_calls": self._api_call_count,
                        "finish_reason": r.get("finish_reason"),
                    })
            except TimeoutError:
                teacher_solvable[i] = False
                teacher_answers[i] = None
                teacher_raw_responses[i] = "[timeout]"
            except Exception as e:
                teacher_solvable[i] = False
                teacher_answers[i] = None
                teacher_raw_responses[i] = f"[error]{e}"

        # --- Update global teacher-difficulty running average over solvable items ---
        try:
            import numpy as _np
            vals = [float(d) for d, s in zip(teacher_difficulty, teacher_solvable)
                    if s and _np.isfinite(d)]
            if vals:
                with self.trainer_ref._metrics_lock:
                    m = self.trainer_ref._augmentation_metrics
                    cnt_prev = int(m.get("teacher_difficulty_count", 0))
                    avg_prev = float(m.get("avg_teacher_difficulty", 0.0))
                    cnt_add = len(vals)
                    m["avg_teacher_difficulty"] = (
                        avg_prev * cnt_prev + float(sum(vals))) / (cnt_prev + cnt_add)
                    m["teacher_difficulty_count"] = cnt_prev + cnt_add
        except Exception as e:
            logger.debug(f"teacher difficulty metric update failed: {e}")

        # Attach teacher results into DataProto
        sub_proto.non_tensor_batch["teacher/gt"] = teacher_answers
        sub_proto.non_tensor_batch["teacher/solvable"] = np.array(
            teacher_solvable, dtype=bool)
        sub_proto.non_tensor_batch["teacher/raw_responses"] = teacher_raw_responses

        clean_questions = [(c or "") if isinstance(
            c, str) else "" for c in clean_questions]

        # === Write cleaned question + metadata back into the proto ===
        # 1) Overwrite raw text with the cleaned question (so downstream uses the fixed wording)
        sub_proto.non_tensor_batch["raw_prompt_data"] = np.asarray(
            clean_questions, dtype=object)
        sub_proto.non_tensor_batch["teacher/difficulty"] = np.asarray(
            teacher_difficulty, dtype=float)

        # 2) Retokenize the cleaned questions so tensors match the cleaned text
        try:
            enc = self.trainer_ref._tokenize_texts(clean_questions)
            sub_proto.batch["input_ids"] = enc["input_ids"]
            sub_proto.batch["attention_mask"] = enc["attention_mask"]
            sub_proto.batch["position_ids"] = enc["position_ids"]
        except Exception as e:
            logger.warning(
                f"Retokenization of cleaned questions failed; using old tensors. {e}")

        # 3) Optionally adjust the estimated reward using teacher difficulty (harder → higher weight)
        try:
            # keep a copy BEFORE adjustment so parents can use raw/teacher later
            est_raw = np.asarray(
                sub_proto.non_tensor_batch["policy/est_reward"], dtype=float)
            sub_proto.non_tensor_batch["policy/est_reward_raw"] = est_raw.copy()
            est_arr = est_raw
            d_policy_raw = sub_proto.non_tensor_batch.get(
                "difficulty_factors", [])
            if len(d_policy_raw) == 0:
                d_policy_raw = [1.0] * len(teacher_difficulty)
            d_policy = np.maximum(np.asarray(d_policy_raw, dtype=float), 1e-6)
            # sign-aware scaling by policy difficulty only
            adj = np.where(est_arr >= 0, est_arr /
                           d_policy, est_arr * d_policy)
            sub_proto.non_tensor_batch["policy/est_reward"] = np.clip(
                adj, -1.0, 1.0)

        except Exception as e:
            logger.debug(f"Difficulty-based reward adjust skipped: {e}")

        _sanitize_non_tensor_batch(sub_proto)

        keep_idxs = np.asarray([i for i, ok in enumerate(
            teacher_solvable) if ok], dtype=np.int64)
        if keep_idxs.size == 0:
            logger.debug("No solvable items in batch")
            return

        solvable_proto = sub_proto[keep_idxs]
        self.trainer_ref.submit_teacher_batch(solvable_proto)


# ==============================
# Trainer with improved error handling
# ==============================


class RayDAPOTrainer(RayPPOTrainer):
    """
    PPO + dynamic queue/augmentation with comprehensive logging for offline analysis.
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.teacher_annotator = None
        self._inbox_lock = threading.Lock()
        self._annotated_inbox = deque(maxlen=20000)

        # create an empty pool so other code paths can query it safely
        self.query_pool = None

        self._seed_records_template: List[QueryRecord] = []
        self._reseed_round: int = 0

        # Memory of queries that reached training (to recycle if pool gets small)
        self._trained_archive: Dict[str, QueryRecord] = {}
        self._rng = np.random.default_rng()

        # ---- Refresh mode (string or bool for backward compat) ----
        mode_raw = getattr(self.config.dynamic_data, "refresh_reward", False)
        if mode_raw is True:
            self.refresh_mode = "children_aggregation"
        elif mode_raw in {"children_aggregation", "path_aggregation"}:
            self.refresh_mode = mode_raw
        else:
            self.refresh_mode = None  # disabled

        # optional caps for very bushy subtrees; safe defaults
        self.refresh_path_cap = int(
            getattr(self.config.dynamic_data, "refresh_path_cap", 2000))
        self.refresh_path_sample = bool(
            getattr(self.config.dynamic_data, "refresh_path_sample", True))

        # Initialize tokenizer if not already done
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                model_name = getattr(
                    config.model, 'model_name_or_path', 'gpt2')
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            except Exception as e:
                logger.warning(
                    f"Failed to initialize tokenizer: {e}. Using default.")
                self.tokenizer = None

        # Initialize metrics tracking
        self._augmentation_metrics = {
            "total_augmented": 0,
            "total_teacher_submitted": 0,
            "total_teacher_integrated": 0,
            "augmentation_success_rate": 0.0,
            "avg_difficulty_factor": 0.0,
            "avg_teacher_difficulty": 0.0,
            "teacher_difficulty_count": 0,
        }

        self._metrics_lock = threading.Lock()

        # Initialize augmentation logger for offline analysis
        log_dir = getattr(config.trainer, 'default_local_dir', './logs')
        experiment_name = getattr(
            config.trainer, 'experiment_name', 'dapo_experiment')
        self.augmentation_logger = AugmentationLogger(log_dir, experiment_name)

        # Track augmentation history for analysis
        self.augmentation_history = []
        self.max_history_size = 100000

        # inside RayDAPOTrainer.__init__(...)
        self._aug_delay_steps = int(
            getattr(self.config.dynamic_data, "aug_delay_steps", 0) or 0)
        self._delayed_lock = threading.Lock()
        self._delayed_aug_buffer: List[Tuple[int, List[QueryRecord]]] = []
        self._delayed_items_count = 0

        # track metrics
        self._augmentation_metrics.update({
            # how many teacher-OK items parked in the delay buffer (cumulative)
            "total_teacher_buffered": 0,
            # how many items later released to the pool (cumulative)
            "total_buffer_released": 0,
        })

    def _ensure_lineage_maps(self) -> None:
        """Ensure lineage/index dicts exist."""
        if not hasattr(self, "_inbox_lock"):
            self._inbox_lock = threading.Lock()
        if not hasattr(self, "_record_index"):
            self._record_index: Dict[str, QueryRecord] = {}
        if not hasattr(self, "_parent_to_children"):
            self._parent_to_children: Dict[str, set] = defaultdict(set)
        if not hasattr(self, "_child_to_parent"):
            self._child_to_parent: Dict[str, Optional[str]] = {}

    def _register_records(self, records: List[QueryRecord]) -> int:
        """
        Register new/updated records:
        - indexes each record by id
        - attaches lineage (parent_id -> children)
        - handles parent changes (moves child across parents)
        Returns: number of records processed.
        """
        if not records:
            return 0

        self._ensure_lineage_maps()
        with self._inbox_lock:
            for rec in records:
                # Ensure an id exists
                if not getattr(rec, "record_id", None):
                    rec.record_id = str(uuid.uuid4())

                rid = rec.record_id
                parent_id = (rec.meta or {}).get("parent_id")

                # Upsert into the main index
                self._record_index[rid] = rec

                # If the record already had a different parent, detach from old parent
                old_parent = self._child_to_parent.get(rid)
                if old_parent and old_parent != parent_id:
                    try:
                        self._parent_to_children[old_parent].discard(rid)
                    except Exception:
                        pass  # tolerant if structure wasn't created yet

                # Attach to current parent (if any)
                if parent_id:
                    self._parent_to_children[parent_id].add(rid)
                    self._child_to_parent[rid] = parent_id
                else:
                    # Track explicit None so we know we considered this node
                    self._child_to_parent[rid] = None

        # (Optional) lightweight debug log
        try:
            if self.augmentation_logger:
                self.augmentation_logger.log_pool_snapshot(
                    {"event": "register_records", "count": len(records)},
                    [r.to_dict() for r in records[:10]],
                )
        except Exception:
            pass
        return len(records)

    def _lookup_record(self, record_id: str) -> Optional[QueryRecord]:
        """Lookup by id, checking live index first, then trained archive."""
        self._ensure_lineage_maps()
        with self._inbox_lock:
            rec = self._record_index.get(record_id)
            if rec is not None:
                return rec
            # Fallback to archive (older copies kept after training updates)
            return self._trained_archive.get(record_id)

    def _get_active_reward(self, rec: QueryRecord) -> float | None:
        """Last step's realized scalar if available, else est_reward."""
        import math
        if rec is None:
            return None
        try:
            v = float(rec.reward) if rec.reward is not None else None
            if v is not None and math.isfinite(v):
                return v
        except Exception:
            pass
        try:
            v = float(rec.est_reward) if rec.est_reward is not None else None
            if v is not None and math.isfinite(v):
                return v
        except Exception:
            pass
        return None

    def _get_policy_difficulty(self, rec: QueryRecord) -> float:
        """Positive finite policy difficulty, default 1.0."""
        import math
        d = (rec.meta or {}).get("policy_difficulty", 1.0) if rec else 1.0
        try:
            d = float(d)
            if not (math.isfinite(d) and d > 0.0):
                return 1.0
            return d
        except Exception:
            return 1.0

    @staticmethod
    def _apply_difficulty(val: float, d: float) -> float:
        """Sign-aware scaling: positive→divide, negative→multiply."""
        if val >= 0:
            return val / max(d, 1e-6)
        return val * max(d, 1e-6)

    def _clip01(self, v: float) -> float:
        return float(max(-1.0, min(1.0, v)))

    def _enqueue_delayed_aug_records(self, items: List[QueryRecord]) -> int:
        """
        If delay disabled, insert directly into pool and return count inserted now.
        If delay enabled, buffer them and return 0 (not inserted yet).
        """
        if not items:
            return 0

        if self._aug_delay_steps <= 0:
            self.query_pool.add_many(items)
            # optional: logging
            if self.augmentation_logger:
                self.augmentation_logger.log_pool_snapshot(
                    {"event": "aug_insert_immediate", "count": len(
                        items), "step": getattr(self, "global_steps", 0)},
                    [r.to_dict() for r in items[:10]]
                )
            return len(items)

        release_at = (getattr(self, "global_steps", 0) or 0) + \
            self._aug_delay_steps
        with self._delayed_lock:
            self._delayed_aug_buffer.append((release_at, items))
            self._delayed_items_count += len(items)

        # metrics + optional logging
        self._augmentation_metrics["total_teacher_buffered"] += len(items)
        if self.augmentation_logger:
            self.augmentation_logger.log_pool_snapshot(
                {"event": "aug_buffer_enqueue", "release_step": release_at,
                    "count": len(items), "step": getattr(self, "global_steps", 0)},
                [r.to_dict() for r in items[:10]]
            )
        return 0

    def _release_due_augmented_to_pool(self) -> int:
        """Move any buffered augmented items whose release_step <= global_steps into the pool."""
        if self.query_pool is None or self._aug_delay_steps <= 0:
            return 0

        now = getattr(self, "global_steps", 0) or 0
        to_release: List[QueryRecord] = []
        with self._delayed_lock:
            keep = []
            for step, items in self._delayed_aug_buffer:
                if step <= now:
                    to_release.extend(items)
                    self._delayed_items_count -= len(items)
                else:
                    keep.append((step, items))
            self._delayed_aug_buffer = keep

        if to_release:
            self.query_pool.add_many(to_release)
            self._augmentation_metrics["total_buffer_released"] += len(
                to_release)
            if self.augmentation_logger:
                self.augmentation_logger.log_pool_snapshot(
                    {"event": "aug_buffer_release", "released": len(
                        to_release), "remaining": self._delayed_items_count, "step": now},
                    [r.to_dict() for r in to_release[:10]]
                )
        return len(to_release)

    def _delayed_buffer_metrics(self) -> Dict[str, Any]:
        with self._delayed_lock:
            bucket_count = len(self._delayed_aug_buffer)
            item_count = self._delayed_items_count
            next_release = min(
                (s for s, _ in self._delayed_aug_buffer), default=None)
        return {
            "delayed_buffer/buckets": bucket_count,
            "delayed_buffer/items": item_count,
            "delayed_buffer/next_release_step": (next_release if next_release is not None else -1),
        }

    def _snapshot_lineage_and_records(self):
        """
        Take a consistent snapshot of lineage maps and the current record lookup.
        Returns:
        parent_to_children: dict[str, set[str]]
        records: dict[str, QueryRecord|None]
        """
        with self._inbox_lock:
            # Safely handle cases where these aren’t created yet
            p2c = getattr(self, "_parent_to_children", None) or {}
            parent_to_children = {p: set(cs) for p, cs in p2c.items()}

            rec_index = getattr(self, "_record_index", None) or {}
            ids = set(rec_index.keys()) | set(self._trained_archive.keys())

            # Fallback to archive if _lookup_record isn’t present or returns None
            def _lk(rid):
                try:
                    r = self._lookup_record(rid)
                except AttributeError:
                    r = None
                return r or self._trained_archive.get(rid)

            records = {rid: _lk(rid) for rid in ids}

        return parent_to_children, records

    def _build_levels(self):
        """
        Returns:
        levels: dict[int, list[str]] mapping depth (0=leaves) -> node_ids
        records: dict[str, QueryRecord|None]
        parent_to_children: dict[str, set[str]]
        Depth is computed as max distance to any leaf (bottom-up).
        """
        parent_to_children, records = self._snapshot_lineage_and_records()

        # gather all nodes
        nodes = set(records.keys()) | set(parent_to_children.keys())
        for cs in parent_to_children.values():
            nodes.update(cs)

        # compute depth: leaves=0, parent depth = 1 + max(children depth)
        depth = {u: 0 for u in nodes}

        # simple DP with repeated relaxation until convergence (trees converge quickly)
        changed = True
        while changed:
            changed = False
            for u in nodes:
                cs = parent_to_children.get(u, ())
                if not cs:
                    continue
                m = max(depth.get(c, 0) for c in cs) + 1
                if m != depth.get(u, 0):
                    depth[u] = m
                    changed = True

        # group by levels
        levels = {}
        for u, d in depth.items():
            levels.setdefault(d, []).append(u)

        return levels, records, parent_to_children

    def _refresh_topo_children_aggregation(self) -> dict[str, float]:
        """
        For each internal node U (bottom → up):
        children_mean = mean( apply_difficulty(child_value, child_policy_difficulty) for child in children(U) )
        new_U = 0.5 * old_U + 0.5 * children_mean  (if old_U missing → just children_mean)
        Returns mapping record_id -> new_value ([-1,1] clipped) for nodes updated.
        """
        levels, records, p2c = self._build_levels()
        out: dict[str, float] = {}
        if not levels:
            return out

        max_depth = max(levels.keys())
        # process depth 1..max (leaves at 0 have no update)
        for d in range(1, max_depth + 1):
            for u in levels.get(d, []):
                cs = list(p2c.get(u, ()))
                if not cs:
                    continue

                # collect children's (already-updated-this-pass or original) values
                contribs = []
                for c in cs:
                    rec_c = records.get(c)
                    # prefer a just-updated value if that child is not a leaf and was updated earlier this pass
                    base_val = out.get(c, self._get_active_reward(rec_c))
                    if base_val is None:
                        continue
                    dpol = self._get_policy_difficulty(rec_c)
                    contribs.append(self._apply_difficulty(base_val, dpol))

                if not contribs:
                    continue

                children_mean = float(sum(contribs) / len(contribs))

                rec_u = records.get(u)
                old_u = self._get_active_reward(rec_u)
                if old_u is None:
                    new_u = children_mean
                else:
                    new_u = 0.5 * old_u + 0.5 * children_mean

                out[u] = self._clip01(new_u)

        return out

    def _enumerate_paths(self, root: str, p2c: dict[str, set[str]]) -> list[list[str]]:
        """
        Returns list of paths from root down to each leaf: [root, ..., leaf].
        WARNING: can be large; we cap/sample per config.
        """
        stack = [(root, [root])]
        paths = []
        while stack:
            u, path = stack.pop()
            cs = list(p2c.get(u, ()))
            if not cs:
                paths.append(path)
                continue
            for v in cs:
                stack.append((v, path + [v]))

            # Cap to avoid blow-up
            if self.refresh_path_sample and len(paths) > self.refresh_path_cap:
                break

        # Optionally sample if too many
        if len(paths) > self.refresh_path_cap and self.refresh_path_sample:
            import random
            random.shuffle(paths)
            paths = paths[:self.refresh_path_cap]
        return paths

    def _refresh_topo_path_aggregation(self) -> dict[str, float]:
        """
        Weighted-children aggregation (no explicit path enumeration).

        Idea:
        - Compute each node's descendant-leaf path count (bottom-up DP).
        - For an internal node U, aggregate child values using weights derived
            from each child's leaf-count, with difficulty-aware adjustment.
        - Blend with the parent's current value:
                new_U = 0.5 * old_U + 0.5 * weighted_mean
            (or weighted_mean if the parent has no current value).
        - Clip refreshed values to [-1, 1].

        Config knobs (optional):
        - config.dynamic_data.path_weighting: "softmax" (default) or "linear"
        - config.dynamic_data.path_weight_temp: float > 0 (default 1.0)
            (used only for "softmax"; lower = sharper)
        """
        import math
        import numpy as _np

        levels, records, p2c = self._build_levels()
        out: dict[str, float] = {}
        if not levels:
            return out

        # --- config ---
        dyn = getattr(self.config, "dynamic_data", {})
        weighting = str(getattr(dyn, "path_weighting", "softmax")
                        ).lower()  # "softmax" | "linear"
        temp = float(getattr(dyn, "path_weight_temp", 1.0))
        temp = max(temp, 1e-6)

        # --- bottom-up leaf path counts ---
        # leaves have count 1; internal nodes sum children counts
        leaf_counts: dict[str, int] = {}
        max_depth = max(levels.keys())

        for u in levels.get(0, []):
            leaf_counts[u] = 1

        for d in range(1, max_depth + 1):
            for u in levels.get(d, []):
                cs = list(p2c.get(u, ()))
                if not cs:
                    # treat as leaf if no children registered
                    leaf_counts[u] = 1
                else:
                    leaf_counts[u] = int(
                        sum(max(1, leaf_counts.get(c, 1)) for c in cs))

        # --- bottom-up refresh using weighted children means ---
        for d in range(1, max_depth + 1):
            for u in levels.get(d, []):
                cs = list(p2c.get(u, ()))
                if not cs:
                    continue

                adj_vals: list[float] = []
                weights_raw: list[float] = []

                for c in cs:
                    rec_c = records.get(c)
                    # Prefer already-updated child value from this pass; else active reward
                    v_c = out.get(c, self._get_active_reward(rec_c))
                    if v_c is None:
                        continue

                    # Difficulty-aware contribution from child
                    v_c_adj = self._apply_difficulty(
                        v_c, self._get_policy_difficulty(rec_c))
                    adj_vals.append(float(v_c_adj))

                    count_c = float(max(leaf_counts.get(c, 1), 1))
                    if weighting == "linear":
                        weights_raw.append(count_c)
                    else:
                        # "softmax": use log(count) / temp for stability and tunable sharpness
                        weights_raw.append(math.log(count_c) / temp)

                if not adj_vals:
                    continue

                # Normalize weights
                if weighting == "linear":
                    w_sum = float(sum(weights_raw))
                    if w_sum <= 0:
                        weights = [1.0 / len(adj_vals)] * len(adj_vals)
                    else:
                        weights = [w / w_sum for w in weights_raw]
                else:
                    m = max(weights_raw) if weights_raw else 0.0
                    exps = [_np.exp(w - m) for w in weights_raw]
                    z = float(sum(exps))
                    if z <= 0:
                        weights = [1.0 / len(adj_vals)] * len(adj_vals)
                    else:
                        weights = [e / z for e in exps]

                weighted_mean = float(
                    sum(w * v for w, v in zip(weights, adj_vals)))

                # Blend with parent's current value (if present)
                rec_u = records.get(u)
                old_u = self._get_active_reward(rec_u)
                new_u = weighted_mean if old_u is None else (
                    0.5 * old_u + 0.5 * weighted_mean)

                out[u] = self._clip01(new_u)

        return out

    def _refresh_rewards_topo(self, mode: str) -> dict[str, float]:
        """
        mode in {"children_aggregation", "path_aggregation"}.
        Returns record_id -> new_value for nodes updated (internal nodes).
        """
        if mode == "children_aggregation":
            return self._refresh_topo_children_aggregation()
        elif mode == "path_aggregation":
            return self._refresh_topo_path_aggregation()
        else:
            return {}

    def _reinsert_all_trained(self):
        """Recycle trained items back into the pool, with levelized refresh."""
        import math

        if not self._trained_archive or self.query_pool is None:
            return 0

        # NEW: levelized refresh using policy difficulty only
        if self.refresh_mode in {"children_aggregation", "path_aggregation"}:
            refreshed_map = self._refresh_rewards_topo(self.refresh_mode)
        else:
            refreshed_map = {}

        items = list(self._trained_archive.values())
        self._rng.shuffle(items)

        to_add: list[QueryRecord] = []
        dropped_eq1 = 0

        for base in items:
            rec = self.query_pool._copy_record_unlocked(base)

            # apply refreshed value if available
            nv = refreshed_map.get(rec.record_id, None)
            if nv is not None:
                rec.reward = nv
                rec.est_reward = nv

            try:
                if rec.reward is not None and math.isclose(float(rec.reward), 1.0, rel_tol=0.0, abs_tol=1e-6):
                    dropped_eq1 += 1
                    continue
            except Exception:
                pass

            to_add.append(rec)

        before = self.query_pool.size()
        if to_add:
            self.query_pool.add_many(to_add)
        added = self.query_pool.size() - before

        self._augmentation_metrics["recycle/dropped_eq1"] = \
            self._augmentation_metrics.get(
                "recycle/dropped_eq1", 0) + dropped_eq1

        return added

    def _clone_seed_record(self, r: QueryRecord, epoch: int, reseed_round: int) -> QueryRecord:
        reward_val = r.reward if (
            r.reward is not None and np.isfinite(r.reward)) else None
        return QueryRecord(
            raw_prompt_data=(r.raw_prompt_data.copy() if isinstance(
                r.raw_prompt_data, np.ndarray) else r.raw_prompt_data),
            input_ids=(r.input_ids.clone()
                       if r.input_ids is not None else None),
            attention_mask=(r.attention_mask.clone()
                            if r.attention_mask is not None else None),
            position_ids=(r.position_ids.clone()
                          if r.position_ids is not None else None),
            gt=r.gt,
            reward=reward_val,
            est_reward=reward_val,
            meta={**(r.meta or {}), "origin": "seed",
                  "epoch": epoch, "reseed_round": reseed_round},
            original_text=r.original_text,
            augmented_text=None,
            teacher_response=None,
        )

    def _pad_for_concat(self, parts: list[DataProto]) -> list[DataProto]:
        """Right-pad all time/sequence-like tensors so last dim matches across parts."""
        if not parts:
            return parts

        import torch

        # Choose a reference length: max over common sequence-ish keys
        def _seq_len(dp: DataProto) -> int:
            for k in ("token_level_rewards", "token_level_scores",
                      "sequences", "input_ids", "attention_mask"):
                if k in dp.batch and isinstance(dp.batch[k], torch.Tensor) and dp.batch[k].dim() >= 2:
                    return int(dp.batch[k].size(-1))
            # Fallback: first 2D+ tensor
            for v in dp.batch.values():
                if isinstance(v, torch.Tensor) and v.dim() >= 2:
                    return int(v.size(-1))
            return 0

        target_L = max(_seq_len(dp) for dp in parts)
        if target_L <= 0:
            return parts  # nothing to do

        # Which keys look like [B, T, ...] (or [B, T]) and should be aligned?
        seq_like_keys = set()
        for dp in parts:
            for k, v in dp.batch.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 2:
                    seq_like_keys.add(k)

        # Pad in-place (right pad with zeros) up to target_L
        for dp in parts:
            for k in seq_like_keys:
                if k not in dp.batch:
                    continue
                v = dp.batch[k]
                if not (isinstance(v, torch.Tensor) and v.dim() >= 2):
                    continue
                curL = int(v.size(-1))
                if curL == target_L:
                    continue
                if curL < target_L:
                    pad = torch.zeros(*v.shape[:-1], target_L - curL,
                                      dtype=v.dtype, device=v.device)
                    dp.batch[k] = torch.cat([v, pad], dim=-1)
                else:
                    # Shouldn't happen since target_L is the max, but keep it safe.
                    dp.batch[k] = v[..., :target_L]
        return parts

    def _norm_text(self, s: str) -> str:
        s = (s or "")
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    def _text_hash(self, s: str) -> str:
        return hashlib.sha256(self._norm_text(s).encode("utf-8")).hexdigest()

    @staticmethod
    def _clamp_diff(x, lo=0.8, hi=1.2) -> float:
        try:
            v = float(x)
        except Exception:
            return 1.0
        if not np.isfinite(v):
            return 1.0
        return float(np.clip(v, lo, hi))

    @staticmethod
    def _parse_difficulty_strict(block: str, lo: float = 0.75, hi: float = 1.33) -> float:
        if not block:
            return 1.0
        s = str(block).strip()

        # strip simple code fences
        if s.startswith("```") and s.endswith("```"):
            s = re.sub(r"^```(?:\w+)?\s*|\s*```$",
                       "", s, flags=re.DOTALL).strip()

        # normalize comma decimals
        s = re.sub(r'(?<=\d),(?=\d)', '.', s)

        # JSON-like: {"difficulty": 0.97}
        m = re.search(
            r'(?i)"difficulty"\s*:\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))', s)
        if m:
            return RayDAPOTrainer._clamp_diff(m.group(1), lo, hi)

        # Labeled: Difficulty: 1.03  or  diff=0.95
        m = re.search(
            r'(?i)\b(?:difficulty|diff)\s*[:=]\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+))\b', s)
        if m:
            return RayDAPOTrainer._clamp_diff(m.group(1), lo, hi)

        # Naked single number only (no extra words)
        if re.fullmatch(r'\s*[-+]?(?:\d+(?:\.\d+)?|\.\d+)\s*\Z', s):
            return RayDAPOTrainer._clamp_diff(s, lo, hi)

        # Hard-parse failed → unchanged difficulty
        return 1.0

    def _ensure_dataproto(self, batch_dict_or_dp) -> DataProto:
        """
        Ensure we have a DataProto with tokenized tensors and standard non-tensor keys for our pipeline.
        Tries DataProto.from_single_dict, and if tensors are missing, falls back to builder for the dataset format.
        """
        if isinstance(batch_dict_or_dp, DataProto):
            dp = batch_dict_or_dp
        else:
            # Try to parse via DataProto (if caller already collated tensors)
            try:
                dp = DataProto.from_single_dict(batch_dict_or_dp)
            except Exception:
                dp = None

        # If we already have tensors, just standardize non-tensor portions a bit
        if dp is not None and "input_ids" in dp.batch and "attention_mask" in dp.batch:
            # Make sure reward_model exists + aliases align
            self._prepare_reward_model_inputs(dp)
            # Ensure raw_prompt_data exists (needed later)
            if "raw_prompt_data" not in dp.non_tensor_batch:
                # Try to reconstruct from 'prompt' or 'extra_info.raw_problem'
                texts = []
                if "prompt" in dp.non_tensor_batch:
                    for p in list(dp.non_tensor_batch["prompt"]):
                        texts.append(_extract_text_from_prompt_field(p))
                elif "extra_info" in dp.non_tensor_batch:
                    for ei in list(dp.non_tensor_batch["extra_info"]):
                        if isinstance(ei, dict):
                            texts.append(str(ei.get("raw_problem", "")))
                        else:
                            texts.append("")
                else:
                    # Fallback: decode tokens
                    for i in range(len(dp.batch["input_ids"])):
                        try:
                            ids = dp.batch["input_ids"][i].detach(
                            ).cpu().numpy()
                            texts.append(self._decode_tokens_to_text(ids))
                        except Exception:
                            texts.append("")
                dp.non_tensor_batch["raw_prompt_data"] = np.asarray(
                    texts, dtype=object)

            # Provide default driver_reward to avoid key errors when popping
            n = len(dp.batch["input_ids"])
            if "driver_reward" not in dp.non_tensor_batch:
                dp.non_tensor_batch["driver_reward"] = np.full(
                    n, np.nan, dtype=float)

            # Make sure position_ids exist if we have attention_mask
            if "attention_mask" in dp.batch and "position_ids" not in dp.batch:
                dp.batch["position_ids"] = _build_position_ids(
                    dp.batch["attention_mask"])

            _sanitize_non_tensor_batch(dp)
            n = len(dp.batch["input_ids"])
            if "origin" not in dp.non_tensor_batch:
                dp.non_tensor_batch["origin"] = np.array(
                    ["seed"] * n, dtype=object)
            if "is_augmented" not in dp.non_tensor_batch:
                dp.non_tensor_batch["is_augmented"] = np.zeros(n, dtype=bool)
            return dp

        # Build a proper DataProto from the dataset's raw JSON row
        return self._build_dp_from_dataset_batch(batch_dict_or_dp)

    def _build_dp_from_dataset_batch(self, row: Dict[str, Any]) -> DataProto:
        """
        Build a DataProto for a single dataset example (your immutable JSON format).
        - Tokenizes the single 'prompt' into tensors
        - Packs reward_model, ability, data_source, extra_info into non-tensor arrays
        - Provides 'raw_prompt_data' and a default 'driver_reward'
        """
        # Extract text
        text = _extract_text_from_prompt_field(row.get("prompt", "")) or \
            (row.get("extra_info", {}).get("raw_problem", "")
             if isinstance(row.get("extra_info", {}), dict) else "")
        text = str(text)

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        enc = self._tokenize_texts([text])

        # Non-tensor fields
        ability = row.get("ability", None)
        data_source = row.get("data_source", "train")
        extra_info = row.get("extra_info", {})
        reward_model = row.get("reward_model", {})

        # Pack into arrays of length 1
        nt = {
            "raw_prompt_data": np.asarray([text], dtype=object),
            "ability": np.asarray([ability], dtype=object),
            "data_source": np.asarray([data_source], dtype=object),
            "extra_info": np.asarray([extra_info], dtype=object),
            "reward_model": _normalize_reward_model_list(reward_model, 1, extra_info_list=[extra_info]),
            # default base reward for augmentation
            "driver_reward": np.asarray([0.5], dtype=float),
        }

        # Build batch
        td = _to_td({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "position_ids": enc["position_ids"],
        })
        dp = DataProto(batch=td, non_tensor_batch=nt, meta_info={})

        # Create aliases/solvable flags, etc.
        self._prepare_reward_model_inputs(dp)
        _sanitize_non_tensor_batch(dp)
        return dp

    # ----------------------------
    # Helpers: DataProto <-> QueryRecord
    # ----------------------------

    def _seed_records_from_loader(self) -> List[QueryRecord]:
        """Load seed QueryRecords from the training dataloader, robust to raw JSON rows."""
        seed: List[QueryRecord] = []
        seed_cap = getattr(self.config.dynamic_data, "seed_cap", 0) or 0
        cap_left = seed_cap if seed_cap > 0 else float("inf")

        for batch_obj in self.train_dataloader:
            if cap_left <= 0:
                break

            try:
                # Build/ensure a DataProto for this row or pre-batched object
                dp = self._ensure_dataproto(batch_obj)

                if "input_ids" not in dp.batch:
                    logger.warning(
                        "Seed batch missing input_ids after coercion; skipping")
                    continue

                B = len(dp.batch["input_ids"])
                take = int(min(B, cap_left))

                # Pull ground truth aliases, raw text, etc.
                gt_list = list(dp.non_tensor_batch.get("gt", [None] * B))
                raw_texts = list(dp.non_tensor_batch.get(
                    "raw_prompt_data", [""] * B))
                data_srcs = list(dp.non_tensor_batch.get(
                    "data_source", ["train"] * B))
                driver_rewards = list(dp.non_tensor_batch.get(
                    "driver_reward", np.full(B, 0.5)))

                for i in range(take):
                    input_ids = dp.batch["input_ids"][i]
                    attention_mask = dp.batch["attention_mask"][i]
                    pos_ids = dp.batch.get(
                        "position_ids", [None] * B)[i] if "position_ids" in dp.batch else None

                    # Raw "prompt data" — keep token ids as raw source; original_text for readability
                    raw_prompt_data = input_ids.detach().cpu().numpy()
                    original_text = str(raw_texts[i]) if i < len(
                        raw_texts) else None
                    gt_val = gt_list[i] if i < len(gt_list) else None
                    if gt_val is None or str(gt_val).strip() == "":
                        continue  # skip this seed

                    seed.append(
                        QueryRecord(
                            raw_prompt_data=raw_prompt_data,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=pos_ids,
                            gt=gt_val,
                            reward=None,            # NEW: no initial reward
                            est_reward=None,        # NEW: defer
                            original_text=original_text,
                            meta={
                                "source": data_srcs[i] if i < len(data_srcs) else "train",
                                "epoch": 0,
                                "origin": "seed",
                            },
                        )
                    )

                    cap_left -= 1

            except Exception as e:
                logger.error(f"Error loading seed record: {e}", exc_info=True)
                continue

        logger.info(f"Loaded {len(seed)} seed records")

        if self.augmentation_logger and seed:
            self.augmentation_logger.log_pool_snapshot(
                {"seed_records": len(seed), "type": "initial_seed"},
                [r.to_dict() for r in seed[:10]],
            )
        return seed

    def _records_to_dataproto(self, recs: List[QueryRecord]) -> DataProto:
        """Convert records to a uniform-shaped DataProto (pads/truncs to a common length)."""
        if not recs:
            raise ValueError("Cannot convert empty record list to DataProto")

        # Keep only records with tensors
        valid_recs = []
        for i, r in enumerate(recs):
            if r.input_ids is not None and r.attention_mask is not None:
                valid_recs.append(r)
            else:
                logger.warning(
                    f"Record {i} missing required tensor fields, skipping")
        if not valid_recs:
            raise ValueError("No valid records to convert to DataProto")

        # Figure out lengths and target length
        lens = [int(r.input_ids.size(-1)) for r in valid_recs]
        max_len_in_batch = max(lens) if lens else 0
        cfg_max = int(getattr(self.config.data,
                      "max_prompt_length", max_len_in_batch or 1))
        target_len = min(max_len_in_batch, cfg_max)

        truncation_mode = str(
            getattr(self.config.data, "truncation", "left")).lower()
        left_trunc = (truncation_mode == "left")
        tok = getattr(self, "tokenizer", None)
        pad_id = getattr(tok, "pad_token_id", None)
        if pad_id is None and tok is not None:
            # they set pad_token = eos_token at init if missing
            pad_id = tok.eos_token_id
        if pad_id is None:
            pad_id = 0

        padded_ids = []
        padded_mask = []
        padded_pos = []

        # Per-item pad/trunc helper
        def _pad_trunc(ids: torch.Tensor, am: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            L = int(ids.size(-1))
            device = ids.device
            if L > target_len:
                if left_trunc:
                    ids2 = ids[..., L - target_len:]
                    am2 = am[...,  L - target_len:]
                else:
                    ids2 = ids[..., :target_len]
                    am2 = am[..., :target_len]
            elif L < target_len:
                pad_n = target_len - L
                if left_trunc:  # left pad
                    ids2 = torch.cat(
                        [torch.full((pad_n,), pad_id, dtype=ids.dtype, device=device), ids], dim=-1)
                    am2 = torch.cat(
                        [torch.zeros((pad_n,), dtype=am.dtype, device=device), am], dim=-1)
                else:  # right pad
                    ids2 = torch.cat(
                        [ids, torch.full((pad_n,), pad_id, dtype=ids.dtype, device=device)], dim=-1)
                    am2 = torch.cat(
                        [am, torch.zeros((pad_n,), dtype=am.dtype, device=device)], dim=-1)
            else:
                ids2, am2 = ids, am
            return ids2, am2

        for r in valid_recs:
            ids = r.input_ids
            am = r.attention_mask if r.attention_mask is not None else torch.ones_like(
                ids, dtype=torch.long)
            # ensure 1D last dim
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
                am = am.unsqueeze(0)
            ids2, am2 = _pad_trunc(ids.squeeze(0), am.squeeze(0))
            # recompute position_ids from mask to stay consistent
            pos2 = _build_position_ids(am2)
            padded_ids.append(ids2)
            padded_mask.append(am2)
            padded_pos.append(pos2)

        input_ids = torch.stack(padded_ids, dim=0)
        attention_mask = torch.stack(padded_mask, dim=0)
        position_ids = torch.stack(padded_pos, dim=0)

        # Non-tensor metadata
        nt = {
            "raw_prompt_data": np.array([r.raw_prompt_data for r in valid_recs], dtype=object),
            "driver_reward":   np.array([(float(r.reward) if r.reward is not None and np.isfinite(r.reward) else np.nan) for r in valid_recs], dtype=np.float32),
            "driver_est_reward": np.array(
                [(float(r.est_reward) if r.est_reward is not None and np.isfinite(
                    r.est_reward) else np.nan) for r in valid_recs],
                dtype=np.float32
            ),
            "driver_gt":     np.array([r.gt for r in valid_recs], dtype=object),
            "record_ids":    np.array([r.record_id for r in valid_recs], dtype=object),
            "original_text": np.array([r.original_text for r in valid_recs], dtype=object),
            "data_source": np.array([(r.meta or {}).get("source", "train") for r in valid_recs], dtype=object),
            "origin":      np.array([(r.meta or {}).get("origin", "seed") for r in valid_recs], dtype=object),
            "is_augmented": np.array([((r.meta or {}).get("origin") == "augmented") for r in valid_recs], dtype=bool),
            "seq_len":     np.array(lens, dtype=np.int32),
        }

        nt["reward_model"] = np.asarray(
            [{"ground_truth": r.gt, "style": None} for r in valid_recs],
            dtype=object
        )

        batch = {"input_ids": input_ids,
                 "attention_mask": attention_mask, "position_ids": position_ids}
        return DataProto(batch=_to_td(batch), non_tensor_batch=nt, meta_info={})

    def _proto_to_query_records(self, dp: DataProto) -> List[QueryRecord]:
        """Convert DataProto to QueryRecords with comprehensive tracking."""
        recs: List[QueryRecord] = []

        if "input_ids" not in dp.batch:
            logger.warning("DataProto missing input_ids")
            return recs

        n = len(dp.batch["input_ids"])

        # Get arrays with proper bounds checking and defaults
        def safe_get_array(key, default_val=None, expected_len=n):
            arr = dp.non_tensor_batch.get(key, [])
            if not isinstance(arr, (list, np.ndarray)):
                arr = []
            arr = list(arr)
            # Pad or truncate to expected length
            if len(arr) < expected_len:
                arr.extend([default_val] * (expected_len - len(arr)))
            elif len(arr) > expected_len:
                arr = arr[:expected_len]
            return arr

        gt_list = safe_get_array("teacher/gt", None)
        est_list = safe_get_array("policy/est_reward", None)
        solv_list = safe_get_array("teacher/solvable", True)
        original_texts = safe_get_array("original_text", None)
        augmented_texts = safe_get_array("raw_prompt_data", None)
        teacher_responses = safe_get_array("teacher/raw_responses", None)
        teacher_diffs = safe_get_array("teacher/difficulty", 1.0)
        parent_ids = safe_get_array("parent_record_id", None)
        record_ids = safe_get_array("record_ids", None)
        est_raw_list = safe_get_array("policy/est_reward_raw", None)
        policy_diffs = safe_get_array("difficulty_factors", 1.0)

        for i in range(n):
            # Respect solvability flag
            if not bool(solv_list[i]):
                # Still log unsolvable items for analysis
                if self.augmentation_logger:
                    self.augmentation_logger.log_annotation({
                        "original_text": str(original_texts[i]) if original_texts[i] else None,
                        "augmented_text": str(augmented_texts[i]) if augmented_texts[i] else None,
                        "solvable": False,
                        "reason": "marked_unsolvable_by_teacher",
                        "estimated_reward": float(est_list[i]) if est_list[i] is not None else None,
                    })
                continue

            token_ids_np = dp.batch["input_ids"][i].detach().cpu().numpy()

            # Safe reward extraction
            try:
                est_val = float(
                    est_list[i]) if est_list[i] is not None else 0.5
                final_reward = np.clip(
                    est_val, -1.0, 1.0) if np.isfinite(est_val) else 0.5
            except (TypeError, ValueError) as e:
                logger.debug(f"Invalid reward value at index {i}: {e}")
                final_reward = 0.5

            # Decode augmented text if needed
            augmented_text_str = None
            if augmented_texts[i] is not None:
                if isinstance(augmented_texts[i], str):
                    augmented_text_str = augmented_texts[i]
                else:
                    try:
                        augmented_text_str = self._decode_tokens_to_text(
                            np.asarray(augmented_texts[i]))
                    except Exception as e:
                        logger.debug(f"Failed to decode augmented text: {e}")
                        augmented_text_str = str(augmented_texts[i])

            # Get position_ids safely
            position_ids = None
            if "position_ids" in dp.batch and i < len(dp.batch["position_ids"]):
                position_ids = dp.batch["position_ids"][i]

            record = QueryRecord(
                raw_prompt_data=token_ids_np,
                input_ids=dp.batch["input_ids"][i],
                attention_mask=dp.batch["attention_mask"][i],
                position_ids=position_ids,
                gt=gt_list[i],
                reward=final_reward,
                est_reward=final_reward,
                original_text=str(
                    original_texts[i]) if original_texts[i] else None,
                augmented_text=augmented_text_str,
                teacher_response=str(
                    teacher_responses[i]) if teacher_responses[i] else None,
                record_id=(
                    str(record_ids[i]) if record_ids[i] else str(uuid.uuid4())),
                meta={
                    "source": "math_dapo",
                    "origin": "augmented",
                    "solvable": bool(solv_list[i]),
                    "global_step": getattr(self, 'global_steps', 0),
                    "teacher_difficulty": float(teacher_diffs[i]) if teacher_diffs[i] is not None else 1.0,
                    "parent_id": str(parent_ids[i]) if parent_ids[i] else None,
                    "est_reward_raw": (float(est_raw_list[i]) if est_raw_list[i] is not None else None),
                    "policy_difficulty": (float(policy_diffs[i]) if policy_diffs[i] is not None else 1.0),
                },
            )
            recs.append(record)

        return recs

    # ----------------------------
    # Helpers: text encode/decode
    # ----------------------------
    def _decode_tokens_to_text(self, token_seq: np.ndarray) -> str:
        """Decode tokens with proper error handling."""
        try:
            if isinstance(token_seq, (str, bytes)):
                return token_seq if isinstance(token_seq, str) else token_seq.decode("utf-8", errors="ignore")
            if self.tokenizer is not None:
                return self.tokenizer.decode(list(token_seq), skip_special_tokens=True)
            else:
                logger.warning(
                    "Tokenizer not initialized, returning string representation")
                return str(token_seq)
        except Exception as e:
            logger.warning(f"Failed to decode tokens: {e}")
            return str(token_seq)

    def _tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize texts with validation and include position_ids."""
        if not texts:
            raise ValueError("Cannot tokenize empty text list")

        max_len = int(getattr(self.config.data, "max_prompt_length", 2048))
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        position_ids = _build_position_ids(attention_mask)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    # ----------------------------
    # Policy-driven augmentation with comprehensive logging
    # ----------------------------

    def generate_augmented_queries(
        self,
        source_batch: DataProto,
        num_per_prompt: int = 1,
        aug_cfg: Optional[Dict] = None,
    ) -> DataProto:
        """Generate augmented queries with logging; returns a DataProto ready for teacher annotation."""
        import numpy as _np
        import torch as _torch

        # Allow caller to override base rewards per original (by normalized text hash)
        overrides_hash = None
        if aug_cfg and isinstance(aug_cfg, dict):
            ov = aug_cfg.get("base_reward_overrides_hash")
            if isinstance(ov, dict) and ov:
                overrides_hash = ov

        # Local helper to wrap plain dict -> TensorDict with a 1D batch_size
        try:
            from tensordict import TensorDict as _TD
        except Exception:
            from torchrl.data import TensorDict as _TD

        def _to_td(_d: Dict[str, _torch.Tensor]) -> _TD:
            # Ensure attention_mask exists before position_ids
            if "attention_mask" in _d and "position_ids" not in _d:
                am = _d["attention_mask"]
                try:
                    # Safe even for 1D / empty, per the new _build_position_ids
                    _d["position_ids"] = _build_position_ids(am)
                except Exception:
                    # Fall back to a zeros-like tensor to avoid crashing
                    _d["position_ids"] = torch.zeros_like(am, dtype=torch.long)
            B = 0
            for v in _d.values():
                if isinstance(v, _torch.Tensor):
                    B = int(v.size(0)) if v.dim() > 0 else 0
                    break
            return _TD(_d, batch_size=[B])

        # Handle disabled/zero case
        if num_per_prompt <= 0:
            empty_td = _to_td({
                "input_ids": _torch.zeros((0, 0), dtype=_torch.long),
                "attention_mask": _torch.zeros((0, 0), dtype=_torch.long),
            })
            return DataProto(
                batch=empty_td,
                non_tensor_batch={
                    "raw_prompt_data": _np.array([], dtype=object),
                    "policy/est_reward": _np.array([], dtype=float),
                },
                meta_info={},
            )

        try:
            gen_start = time.time()

            # 1) Gather unique originals + base rewards
            raw_data_list = list(
                source_batch.non_tensor_batch.get(
                    "raw_prompt_data",
                    source_batch.non_tensor_batch.get("raw_prompt_ids", []),
                )
            )
            base_rewards_arr = list(
                source_batch.non_tensor_batch.get(
                    "driver_reward",
                    _np.full(len(raw_data_list), 0.5),
                )
            )

            key2idx: Dict[Tuple[int, ...], int] = {}
            originals: List[_np.ndarray] = []
            original_texts: List[str] = []
            base_rewards_per_original: List[float] = []

            for i, arr in enumerate(raw_data_list):
                # Build a stable key even if arr is text
                try:
                    key = tuple(_np.asarray(arr, dtype=int).tolist())
                except (TypeError, ValueError):
                    try:
                        key = tuple(map(ord, str(arr)))
                    except Exception:
                        key = (i,)

                if key not in key2idx:
                    key2idx[key] = len(originals)
                    arr_np = _np.asarray(arr)
                    originals.append(arr_np)
                    original_texts.append(self._decode_tokens_to_text(arr_np))
                    br = float(base_rewards_arr[i]) if i < len(
                        base_rewards_arr) else 0.5
                    br = float(_np.clip(br, -1.0, 1.0)
                               ) if _np.isfinite(br) else 0.5
                    base_rewards_per_original.append(br)

            # Extract parent record_ids from the source batch (if present)
            parent_rids_src = list(
                source_batch.non_tensor_batch.get("record_ids", []))

            # Map each unique ORIGINAL key to its first-seen parent record_id
            key2parent: Dict[Tuple[int, ...], str] = {}
            for i, arr in enumerate(raw_data_list):
                try:
                    key = tuple(_np.asarray(arr, dtype=int).tolist())
                except (TypeError, ValueError):
                    try:
                        key = tuple(map(ord, str(arr)))
                    except Exception:
                        key = (i,)
                if key not in key2parent:
                    pid = str(parent_rids_src[i]) if i < len(
                        parent_rids_src) else None
                    key2parent[key] = pid

            # Build a list aligned to "originals"
            original_parent_ids: List[Optional[str]] = []
            for i, arr in enumerate(originals):
                # reconstruct the same key logic
                try:
                    key = tuple(_np.asarray(arr, dtype=int).tolist())
                except Exception:
                    try:
                        key = tuple(map(ord, str(original_texts[i])))
                    except Exception:
                        key = (i,)
                original_parent_ids.append(key2parent.get(key))

            # ---- use actual per-original rewards when provided ----
            if overrides_hash:
                for idx, txt in enumerate(original_texts):
                    key = self._text_hash(txt)
                    if key in overrides_hash:
                        base_rewards_per_original[idx] = float(
                            np.clip(overrides_hash[key], -1.0, 1.0))

            # ==== Anchors (non-codey, plain-text) ====
            ORIG_TAG = "<ORIGINAL>"
            NEW_TAG = "<NEW>"
            DIFF_TAG = "<DIFFICULTY>"
            END_TAG = "<END>"

            def _escape_braces(s: str) -> str:
                return s.replace("{", "{{").replace("}", "}}")

            # Small-model-friendly instructions: numeric-only tweaks, plain text, no code.
            parts = []
            parts.append(
                "You are an expert math problem writer.\n"
                "Given an original problem, produce a NEW problem that is similar by changing ONLY numeric values "
                "(constants, coefficients, exponents, lengths, bounds) by small amounts. Keep topic/structure/variables/target the same.\n"
                "Do NOT output any code or solutions to the problem. Do not generate any rationale or explanations.\n"
                "Follow this format exactly (with two examples below) to complete the prompt, with no extra text.\n"
                f"{ORIG_TAG}\n"
                "the original problem text only (no answer)\n"
                f"{NEW_TAG}\n"
                "your minimally edited problem text only (no answer)\n"
                f"{DIFF_TAG}\n"
                "a single number like 0.98 indicating NEW vs original difficulty\n"
                f"{END_TAG}\n"
            )

            # ==== Example 2 (slightly harder) ====
            ex3_original = (
                "In $\\triangle ABC$, we have $AC=BC=7$ and $AB=2$. Suppose that $D$ is a point "
                "on the line $AB$ such that $B$ lies between $A$ and $D$, and $CD=8$. "
                "What is the length of the segment $BD$?"
            )
            ex3_new = (
                "In $\\triangle ABC$, we have $AC=BC=8$ and $AB=3$. Suppose that $D$ is a point "
                "on the line $AB$ such that $B$ lies between $A$ and $D$, and $CD=9$. "
                "What is the length of the segment $BD$?"
            )
            ex3_diff = 1.1

            parts += [
                f"{ORIG_TAG}\n", _escape_braces(ex3_original) + "\n",
                f"{NEW_TAG}\n",  _escape_braces(ex3_new) + "\n",
                f"{DIFF_TAG}\n", str(ex3_diff) + "\n",
                f"{END_TAG}\n",
            ]

            # ==== Your turn ====
            parts += [
                f"{ORIG_TAG}\n",
                "{original_problem}\n\n",
                f"{NEW_TAG}\n"
            ]

            TEMPLATE = "".join(parts)

            aug_prompts: List[str] = []
            base_rewards_for_aug: List[float] = []
            original_texts_for_aug: List[str] = []
            original_indices: List[int] = []
            parent_ids_for_aug: List[Optional[str]] = []

            for idx, br in enumerate(base_rewards_per_original):
                orig_text = original_texts[idx] if idx < len(
                    original_texts) else ""
                if "is the answer to the problem." in orig_text:
                    orig_text = orig_text.split(
                        "is the answer to the problem.")[1]
                if "Remember to put your answer" in orig_text:
                    orig_text = orig_text.split(
                        "Remember to put your answer")[0]
                for _ in range(num_per_prompt):
                    orig_text_esc = _escape_braces(orig_text.strip())
                    aug_prompts.append(TEMPLATE.format(
                        original_problem=orig_text_esc))
                    base_rewards_for_aug.append(br)
                    original_texts_for_aug.append(orig_text.strip())
                    original_indices.append(idx)
                    parent_ids_for_aug.append(
                        original_parent_ids[idx] if idx < len(original_parent_ids) else None)

            if not aug_prompts:
                empty_td = _to_td({
                    "input_ids": _torch.zeros((0, 0), dtype=_torch.long),
                    "attention_mask": _torch.zeros((0, 0), dtype=_torch.long),
                })
                return DataProto(
                    batch=empty_td,
                    non_tensor_batch={
                        "raw_prompt_data": _np.array([], dtype=object),
                        "policy/est_reward": _np.array([], dtype=float),
                    },
                    meta_info={},
                )

            enc = self._tokenize_texts(aug_prompts)
            aug_td = _to_td(
                {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "position_ids": enc["position_ids"]})
            aug_batch = DataProto(
                batch=aug_td,
                non_tensor_batch={
                    "aug_prompt_len": enc["attention_mask"].sum(dim=1).cpu().numpy()},
                meta_info={},
            )

            if "position_ids" not in aug_batch.batch and "attention_mask" in aug_batch.batch:
                aug_batch.batch["position_ids"] = _build_position_ids(
                    aug_batch.batch["attention_mask"])

            # === Make aug_batch length divisible by dp_size (worker group size) ===
            dp_size = (
                getattr(self.actor_rollout_wg, "world_size", None)
                or getattr(self.actor_rollout_wg, "size", None)
                or getattr(self.actor_rollout_wg, "n_workers", None)
                or getattr(self.actor_rollout_wg, "num_workers", None)
                or 1
            )
            if dp_size > 1:
                try:
                    N = len(aug_batch.batch["input_ids"])
                except Exception:
                    N = 0
                rem = N % dp_size
                if rem != 0:
                    keep = N - rem
                    if keep <= 0:
                        # nothing to send; return clean empty DP
                        return DataProto(
                            batch=_to_td({
                                "input_ids": _torch.zeros((0, 0), dtype=_torch.long),
                                "attention_mask": _torch.zeros((0, 0), dtype=_torch.long),
                            }),
                            non_tensor_batch={
                                "raw_prompt_data": _np.array([], dtype=object),
                                "policy/est_reward": _np.array([], dtype=float),
                            },
                            meta_info={
                                "augmentation_time": time.time() - gen_start},
                        )
                    aug_batch = aug_batch[:keep]

            # 3) Single generation call (with stopping token added)
            aug_batch.meta_info.update({
                "stop_strings": [
                    # New anchors
                    "<END>", "\n<END>",
                    "<NEW>", "\n<NEW>",
                    "<DIFFICULTY>", "\n<DIFFICULTY>",
                    # Hard stops that usually precede code or solutions
                    "```",  # code fence
                    "Solution:", "\nSolution:",
                    "Answer:", "\nAnswer:",
                    "Proof:", "\nProof:",
                    "\\boxed",
                    "Step 1", "\nStep 1",
                    "To solve", "\nTo solve",
                    "To compute", "\nTo compute",
                    "To calculate", "\nTo calculate",
                    "To find", "\nTo find",

                ],
                "max_new_tokens": 300,
            })

            gen_out = self.actor_rollout_wg.generate_sequences(aug_batch)
            seq_tensor = gen_out.batch.get(
                "sequences", gen_out.batch.get("input_ids", None))
            if seq_tensor is None:
                raise RuntimeError(
                    "generate_sequences did not return 'sequences' or 'input_ids'.")

            # 4) Decode & parse sections
            prompt_lens = aug_batch.non_tensor_batch["aug_prompt_len"]
            new_texts: List[str] = []
            diff_texts: List[str] = []
            full_generations: List[str] = []

            def _extract_sections(gen_txt: str) -> Tuple[str, str]:
                t = (gen_txt or "").strip()
                # Strip simple code fences quickly
                if t.startswith("```") and t.endswith("```"):
                    t = re.sub(r"^```(?:\w+)?\s*|\s*```$",
                               "", t, flags=re.DOTALL).strip()
                t = t.replace("\r\n", "\n")

                # New anchors first
                ORIG_TAG = "<ORIGINAL>"
                NEW_TAG = "<NEW>"
                DIFF_TAG = "<DIFFICULTY>"
                END_TAG = "<END>"

                def _first_span(text, start_pat, end_pat):
                    m_start = re.search(start_pat, text, flags=re.IGNORECASE)
                    if not m_start:
                        return None, None
                    start = m_start.end()
                    m_end = re.search(
                        end_pat, text[start:], flags=re.IGNORECASE)
                    end = (start + m_end.start()) if m_end else len(text)
                    return start, end

                # Try new tags
                s_new, e_new = _first_span(
                    t, r"<\s*NEW\s*>", r"<\s*(?:DIFFICULTY|END|ORIGINAL|NEW)\s*>")
                if s_new is not None:
                    new_problem = t[s_new:e_new].strip()

                    # Difficulty between NEW and next boundary or DIFFICULTY block specifically
                    s_diff, e_diff = _first_span(
                        t, r"<\s*DIFFICULTY\s*>", r"<\s*(?:END|NEW|ORIGINAL)\s*>")
                    diff_src = t[s_diff:e_diff].strip(
                    ) if s_diff is not None else t[e_new:].strip()
                    return new_problem or t, diff_src or t

                # Fallback: legacy #...# headers
                m_new = re.search(r"#\s*New\s*Problem\s*#",
                                  t, flags=re.IGNORECASE)
                if m_new:
                    start = m_new.end()
                    m_next = re.search(
                        r"#\s*(?:Difficulty|End|Original\s*Problem|New\s*Problem)\s*#", t[start:], flags=re.IGNORECASE)
                    end = (start + m_next.start()) if m_next else len(t)
                    new_problem = t[start:end].strip()

                    m_diff = re.search(
                        r"#\s*Difficulty\s*#(?P<body>.*?)(?=#\s*(?:End|New\s*Problem|Original\s*Problem)\s*#|$)",
                        t[end:], flags=re.IGNORECASE | re.DOTALL
                    )
                    diff_src = (m_diff.group("body").strip()
                                if m_diff else t[end:].strip())
                    return new_problem or t, diff_src or t

                # Last resort: return whole text as "new", same as difficulty source
                return t, t

            for i in range(seq_tensor.size(0)):
                seq_ids = seq_tensor[i].detach().cpu().tolist()
                p_len = int(prompt_lens[i]) if i < len(prompt_lens) else 0
                gen_ids = seq_ids[p_len:] if p_len < len(seq_ids) else []
                txt = self.tokenizer.decode(
                    gen_ids, skip_special_tokens=True) if self.tokenizer else str(gen_ids)
                full_generations.append(txt)
                new_prob, diff_src = _extract_sections(txt)
                new_texts.append(new_prob if new_prob else txt)
                diff_texts.append(diff_src)

            # 5) Estimate rewards and log
            diff_factors = [self._parse_difficulty_strict(
                t, lo=0.75, hi=1.33) for t in diff_texts]
            min_len = min(len(base_rewards_for_aug),
                          len(diff_factors), len(new_texts))
            est_rewards: List[float] = []

            # running average of difficulty in metrics
            if diff_factors:
                n = self._augmentation_metrics.get("total_augmented", 0)
                prev = self._augmentation_metrics.get(
                    "avg_difficulty_factor", 1.0)
                window = diff_factors[:min_len]
                self._augmentation_metrics["avg_difficulty_factor"] = (
                    prev * n + sum(window)) / (n + len(window))

            successful_augmentations = []
            for i in range(min_len):
                br = float(base_rewards_for_aug[i])
                df = max(float(diff_factors[i]), 0.01)
                if np.isfinite(df):
                    val = br / df if br >= 0 else br * df
                else:
                    val = br
                val = float(np.clip(val, -1.0, 1.0))
                est_rewards.append(val)

                rec = {
                    "record_id": str(uuid.uuid4()),
                    "original_text": original_texts_for_aug[i] if i < len(original_texts_for_aug) else "",
                    "original_index": original_indices[i] if i < len(original_indices) else 0,
                    "augmented_text": new_texts[i],
                    "full_generation": full_generations[i] if i < len(full_generations) else "",
                    "base_reward": br,
                    "difficulty_factor": df,
                    "difficulty_text": diff_texts[i] if i < len(diff_texts) else "",
                    "estimated_reward": val,
                    "generation_template": "single_pass",
                    "global_step": getattr(self, "global_steps", 0),
                    "epoch": getattr(self, "current_epoch", 0),
                    "text_length_original": len(original_texts_for_aug[i]) if i < len(original_texts_for_aug) else 0,
                    "text_length_augmented": len(new_texts[i]),
                    "generation_time": time.time() - gen_start,
                }
                if self.augmentation_logger:
                    self.augmentation_logger.log_augmentation(rec)
                self.augmentation_history.append(rec)
                if len(self.augmentation_history) > self.max_history_size:
                    self.augmentation_history.pop(0)
                successful_augmentations.append(rec)

            self._augmentation_metrics["total_augmented"] += len(
                new_texts[:min_len])
            self._augmentation_metrics["augmentation_success_rate"] = (
                min_len / len(aug_prompts)) if aug_prompts else 0.0

            # 6) Tokenize NEW problems and build DataProto
            new_enc = self._tokenize_texts(new_texts[:min_len])
            record_ids = [r["record_id"] for r in successful_augmentations]
            nt = {
                "raw_prompt_data": _np.array(new_texts[:min_len], dtype=object),
                "policy/est_reward": _np.array(est_rewards, dtype=float),
                "original_text": _np.array(original_texts_for_aug[:min_len], dtype=object),
                "record_ids": _np.array(record_ids, dtype=object),
                # <-- NEW
                "parent_record_id": _np.array(parent_ids_for_aug[:min_len], dtype=object),
                "difficulty_factors": _np.array(diff_factors[:min_len], dtype=float),
                "data_source": _np.array(["math_dapo"] * min_len, dtype=object),
                "origin": _np.array(["augmented"] * min_len, dtype=object),
                "is_augmented": _np.array([True] * min_len, dtype=bool),
            }

            new_td = _to_td(
                {"input_ids": new_enc["input_ids"], "attention_mask": new_enc["attention_mask"], "position_ids": new_enc["position_ids"]})

            return DataProto(
                batch=new_td,
                non_tensor_batch=nt,
                meta_info={"augmentation_time": time.time() - gen_start},
            )

        except Exception as e:
            logger.error(
                f"Error in generate_augmented_queries: {e}", exc_info=True)
            if self.augmentation_logger:
                self.augmentation_logger.log_augmentation({
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "global_step": getattr(self, "global_steps", 0),
                    "num_per_prompt": num_per_prompt,
                })
            empty_td = _to_td({
                "input_ids": _torch.zeros((0, 0), dtype=_torch.long),
                "attention_mask": _torch.zeros((0, 0), dtype=_torch.long),
            })
            return DataProto(
                batch=empty_td,
                non_tensor_batch={
                    "raw_prompt_data": _np.array([], dtype=object),
                    "policy/est_reward": _np.array([], dtype=float),
                },
                meta_info={},
            )

    # ----------------------------
    # Inbox plumbing for teacher annotations
    # ----------------------------
    def submit_teacher_batch(self, dp: DataProto):
        """Called by AsyncTeacherAnnotator (background thread) with tracking."""
        with self._inbox_lock:
            self._annotated_inbox.append(dp)
            self._augmentation_metrics["total_teacher_submitted"] += len(
                dp.batch.get("input_ids", []))

            # Log submission event
            if self.augmentation_logger:
                record_ids = dp.non_tensor_batch.get("record_ids", [])
                for i in range(min(5, len(record_ids))):  # Log sample
                    self.augmentation_logger.log_annotation({
                        "event": "teacher_batch_submitted",
                        "record_id": str(record_ids[i]) if i < len(record_ids) else None,
                        "batch_size": len(dp.batch.get("input_ids", [])),
                        "global_step": getattr(self, 'global_steps', 0),
                    })

    def _drain_teacher_inbox(self):
        """Main thread: integrate all pending teacher-annotated protos into the pool with logging."""
        packets: List[DataProto] = []
        with self._inbox_lock:
            while self._annotated_inbox:
                packets.append(self._annotated_inbox.popleft())

        if not packets:
            return

        integrated_count = 0
        integrated_records = []

        for dp in packets:
            try:
                new_items: List[QueryRecord] = self._proto_to_query_records(dp)

                # Register in lineage & index before we enqueue/buffer them
                self._register_records(new_items)

                # Ensure rewards are set from est
                ests = dp.non_tensor_batch.get("policy/est_reward", [])
                record_ids = dp.non_tensor_batch.get("record_ids", [])
                est_arr = dp.non_tensor_batch.get("policy/est_reward", [])

                for idx, it in enumerate(new_items):
                    if idx < len(ests):
                        try:
                            reward_val = float(ests[idx])
                            if np.isfinite(reward_val):
                                it.reward = np.clip(reward_val, -1.0, 1.0)
                                it.est_reward = it.reward
                        except (TypeError, ValueError):
                            pass

                    # Track integration
                    integration_record = {
                        "record_id": str(record_ids[idx]) if idx < len(record_ids) else it.record_id,
                        "integrated": True,
                        "final_reward": it.reward,
                        "estimated_reward": float(est_arr[idx]) if idx < len(est_arr) and np.isfinite(est_arr[idx]) else None,
                        "has_gt": it.gt is not None,
                        "global_step": getattr(self, 'global_steps', 0),
                    }
                    integrated_records.append(integration_record)

                inserted_now = self._enqueue_delayed_aug_records(new_items)
                integrated_count += inserted_now  # count only what actually hits the pool now

            except Exception as e:
                logger.error(
                    f"Failed to integrate teacher batch: {e}", exc_info=True)

                # Log integration failure
                if self.augmentation_logger:
                    self.augmentation_logger.log_annotation({
                        "event": "integration_failed",
                        "error": str(e),
                        "batch_size": len(dp.batch.get("input_ids", [])) if dp else 0,
                        "global_step": getattr(self, 'global_steps', 0),
                    })

        self._augmentation_metrics["total_teacher_integrated"] += integrated_count

        # Log successful integrations
        if self.augmentation_logger and integrated_records:
            for record in integrated_records[:10]:  # Log sample
                self.augmentation_logger.log_annotation(record)

    def _prepare_reward_model_inputs(self, dp: DataProto) -> None:
        """
        Populate dp.non_tensor_batch['reward_model'] as an object array of per-item dicts:
            reward_model[i] -> {'ground_truth': <str|None>, 'style': <str|None>, 'solvable': <bool>}
        Also set convenient aliases 'gt' and 'teacher/gt' for components that still read those.
        """
        nt = dp.non_tensor_batch

        # batch length
        try:
            n = len(dp.batch["input_ids"])
        except Exception:
            n = 0

        def _as_list(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (list, tuple)):
                return list(x)
            if x is None:
                return []
            return [x]

        def _broadcast(src, n_):
            lst = _as_list(src)
            if len(lst) == 0:
                return [None] * n_
            if len(lst) == n_:
                return lst
            if n_ % len(lst) == 0:
                rep = n_ // len(lst)
                out = []
                for v in lst:
                    out.extend([v] * rep)
                return out
            if len(lst) < n_:
                return lst + [lst[-1]] * (n_ - len(lst))
            return lst[:n_]

        # 1) Bring in reward_model (dataset-native) if present; make it length-n object array of dicts
        extra_info_list = _broadcast(nt.get("extra_info", None), n)
        if "reward_model" in nt:
            nt["reward_model"] = _normalize_reward_model_list(
                nt["reward_model"], n, extra_info_list=extra_info_list)
        else:
            # Construct from compatible aliases, or fall back to extra_info.raw_answer
            # Try the aliases the old code expected
            gt_candidates = None
            for k in ("teacher/gt", "driver_gt", "gt", "answers", "answer", "final_answer", "labels", "label"):
                if k in nt and len(_as_list(nt[k])) > 0:
                    gt_candidates = _broadcast(nt[k], n)
                    break
            if gt_candidates is None:
                # fallback from extra_info
                gt_candidates = [(ei.get("raw_answer") if isinstance(
                    ei, dict) else None) for ei in extra_info_list]

            rm_list = []
            for i in range(n):
                rm_list.append(
                    {"ground_truth": gt_candidates[i], "style": None})
            nt["reward_model"] = np.asarray(rm_list, dtype=object)

        # 2) Solvability mask (default True)
        solv_list = _broadcast(nt.get("teacher/solvable", True), n)

        def _empty_gt(d):
            if not isinstance(d, dict):
                return True
            gt = d.get("ground_truth", None)
            if gt is None:
                return True
            try:
                import math
                if isinstance(gt, float) and math.isnan(gt):
                    return True
            except Exception:
                pass
            return str(gt).strip() == ""  # blanks

        solv_list = [(bool(solv_list[i]) and not _empty_gt(nt["reward_model"][i]))
                     for i in range(n)]

        # 3) Write back standardized fields
        #    Also expose easy aliases for any components expecting them
        gt_alias = []
        for i in range(n):
            rm = nt["reward_model"][i] if i < len(nt["reward_model"]) else {}
            gt_alias.append((rm or {}).get("ground_truth"))
        nt["gt"] = np.asarray(gt_alias, dtype=object)
        nt["teacher/gt"] = np.asarray(gt_alias, dtype=object)
        nt["teacher/solvable"] = np.asarray(solv_list, dtype=bool)

    # ----------------------------
    # Main training loop with comprehensive logging
    # ----------------------------

    def fit(self):
        """Main training loop with periodic data dumping and analysis."""
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        from verl.utils.profiler import marked_timer

        logger_instance = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 1
        self.gen_steps = 1
        self.current_epoch = 0

        self._load_checkpoint()

        # Dynamic data configuration
        dyn_en = bool(getattr(self.config, "dynamic_data",
                      {}).get("enable", False))

        max_pool_size = int(
            getattr(self.config.dynamic_data, "max_pool_size", 30000))
        self.query_pool = ThreadSafeQueryPool(
            max_size=max_pool_size
        )
        sampling_mode = str(getattr(self.config.dynamic_data,
                            "sampling_mode", "medium_only")).lower()
        self.query_pool.set_mixed_easy_medium(
            sampling_mode in {"mixed_easy_medium", "mixed", "half"})

        # Inbox for background annotations
        self._inbox_lock = threading.Lock()
        self._annotated_inbox: deque[DataProto] = deque(maxlen=20000)

        self.teacher_annotator: Optional[AsyncTeacherAnnotator] = None

        if dyn_en:
            seed_records = self._seed_records_from_loader()
            init_mode = str(getattr(self.config.dynamic_data,
                            "init_mode", "map")).lower()

            if init_mode == "uniform":
                # seed_records already carry tensors; we just overwrite their rewards uniformly
                self.query_pool.initialize_uniform(seed_records)
            else:
                # default: use the mapped/driver rewards built in _seed_records_from_loader()
                self.query_pool.add_many(seed_records)

            # keep a template for future top-ups (fresh clones will be made from these)
            self._seed_records_template = [self._clone_seed_record(
                r, epoch=0, reseed_round=0) for r in seed_records]

            # register seeds for lineage/index before any augmentation happens
            self._register_records(seed_records)

            try:
                # Get model name from config or use default
                model_name = getattr(
                    self.config.dynamic_data, "teacher_model", "gpt-5-mini")
                self.teacher_annotator = AsyncTeacherAnnotator(
                    self,
                    model_name=model_name,
                    augmentation_logger=self.augmentation_logger
                )
                self.teacher_annotator.start()
            except Exception as e:
                logger.error(f"Failed to start teacher annotator: {e}")
                raise

        # Pre-training validation
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger_instance.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._cleanup_teacher_annotator()
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps,
                            initial=self.global_steps, desc="Training Progress")

        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        aug_cfg = getattr(self.config, "augmentation", {})
        do_augment = dyn_en and bool(aug_cfg.get("enable", True))

        # Logging frequencies
        snapshot_frequency = getattr(
            self.config.dynamic_data, "snapshot_frequency", 100)

        try:
            for epoch in range(self.config.trainer.total_epochs):
                self.current_epoch = epoch
                steps_per_epoch = getattr(
                    self.config.trainer, "steps_per_epoch", None)

                if not dyn_en:
                    data_iterable = self.train_dataloader
                else:
                    data_iterable = itertools.count() if steps_per_epoch is None else range(steps_per_epoch)

                for batch_dict in data_iterable:
                    metrics = {}

                    # Drain any teacher results into buffer or pool
                    self._drain_teacher_inbox()

                    # Release buffered augmented items whose delay has expired
                    released_now = self._release_due_augmented_to_pool()

                    # Determine if this is the last step
                    is_last_step = (self.global_steps >=
                                    self.total_training_steps)

                    # Periodic pool snapshot for analysis
                    if dyn_en and self.global_steps % snapshot_frequency == 0:
                        pool_metrics = self.query_pool.get_metrics()
                        pool_sample = self.query_pool.get_sample_for_logging(
                            20)
                        self.augmentation_logger.log_pool_snapshot(
                            pool_metrics, pool_sample)

                        # Log to main metrics
                        for k, v in pool_metrics.items():
                            metrics[f"pool/{k}"] = v

                    # Step 2: sample batch from queue
                    if dyn_en:
                        want = self.config.data.train_batch_size

                        # NEW: recycle trained queries wholesale when pool shrinks
                        if self.query_pool.size() < want:
                            added = self._reinsert_all_trained()
                            logger.info(
                                f"[dynamic] pool<{want}; recycled {added} trained queries")

                        sampled = self.query_pool.sample_batch(k=want)
                        if not sampled:
                            logger.info("[dynamic] queue is empty; waiting…")
                            time.sleep(0.1)
                            continue
                        sampled_records_for_this_step = sampled  # keep for reward update

                        # Log sampling event
                        if self.augmentation_logger and self.global_steps % 10 == 0:
                            self.augmentation_logger.log_pool_snapshot(
                                {"event": "batch_sampled",
                                    "batch_size": len(sampled)},
                                [r.to_dict() for r in sampled[:5]]
                            )

                        new_batch = self._records_to_dataproto(sampled)
                    else:
                        new_batch: DataProto = self._ensure_dataproto(
                            batch_dict)
                        sampled_records_for_this_step = []

                    num_gen_batches += 1

                    # Pop keys for generation
                    pop_keys = ["input_ids", "attention_mask"]
                    if "position_ids" in new_batch.batch:
                        pop_keys.append("position_ids")

                    # Only pop non-tensor keys that exist (avoid DataProto.pop assertion)
                    _wanted_nt = ("raw_prompt_ids", "raw_prompt_data",
                                  "driver_reward", "record_ids")
                    _nt_to_pop = [
                        k for k in _wanted_nt if k in new_batch.non_tensor_batch]

                    gen_batch = new_batch.pop(
                        batch_keys=pop_keys,
                        non_tensor_batch_keys=_nt_to_pop,
                    )

                    # Repeat gen_batch for rollout
                    gen_batch = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True
                    )

                    if "position_ids" not in gen_batch.batch and "attention_mask" in gen_batch.batch:
                        gen_batch.batch["position_ids"] = _build_position_ids(
                            gen_batch.batch["attention_mask"])

                    with marked_timer("step", timing_raw):
                        # Step 3: generate rollouts
                        with marked_timer("gen", timing_raw, "red"):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(
                                gen_batch)
                            timing_raw.update(
                                gen_batch_output.meta_info.get("timing", {}))
                            gen_batch_output.meta_info.pop("timing", None)

                        # REMAX baseline
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            with marked_timer("gen_max", timing_raw, "red"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(
                                    gen_baseline_batch)

                                new_batch = new_batch.union(
                                    gen_baseline_output)
                                reward_baseline_tensor = self.reward_fn(
                                    new_batch)
                                reward_baseline_tensor = reward_baseline_tensor.sum(
                                    dim=-1)
                                new_batch.pop(batch_keys=list(
                                    gen_baseline_output.batch.keys()))
                                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                                del gen_baseline_batch, gen_baseline_output

                        # Correct UID creation
                        n_rep = int(self.config.actor_rollout_ref.rollout.n)
                        gen_bsz = int(gen_batch.batch["input_ids"].size(0))
                        if gen_bsz % n_rep != 0:
                            logger.error(
                                f"Repeated size {gen_bsz} not divisible by n_rep={n_rep}")
                            continue

                        base_bsz = gen_bsz // n_rep
                        new_batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(base_bsz)], dtype=object
                        )

                        new_batch = new_batch.repeat(
                            repeat_times=n_rep, interleave=True)
                        new_batch = new_batch.union(gen_batch_output)

                        # Step 3 (cont.): reward calc
                        with marked_timer("reward", timing_raw, "yellow"):
                            # Prepare & sanitize inputs for reward model
                            self._prepare_reward_model_inputs(new_batch)

                            rm = list(new_batch.non_tensor_batch.get(
                                "reward_model", []))

                            def _missing_gt(d):
                                if not isinstance(d, dict):
                                    return True
                                gt = d.get("ground_truth")
                                if gt is None:
                                    return True
                                try:
                                    import math
                                    if isinstance(gt, float) and math.isnan(gt):
                                        return True
                                except Exception:
                                    pass
                                return str(gt).strip() == ""  # empty

                            keep = [i for i, d in enumerate(
                                rm) if not _missing_gt(d)]
                            if len(keep) != len(rm):
                                if not keep:
                                    # Nothing scorable this step; skip safely
                                    continue
                                new_batch = new_batch[keep]

                            n_items = len(new_batch.batch["input_ids"])
                            if "data_source" not in new_batch.non_tensor_batch:
                                new_batch.non_tensor_batch["data_source"] = np.array(
                                    ["train"] * n_items, dtype=object)
                            else:
                                new_batch.non_tensor_batch["data_source"] = _to_indexable_array(
                                    new_batch.non_tensor_batch["data_source"], n_items
                                )

                            # === Map augmentation source to a supported reward source ===
                            try:
                                ds = list(
                                    new_batch.non_tensor_batch["data_source"])
                                new_batch.non_tensor_batch["data_source"] = np.array(
                                    ["math_dapo" if (str(x) == "augment") else (
                                        x if x else "train") for x in ds],
                                    dtype=object
                                )
                            except Exception:
                                # Last-resort fallback if anything goes odd
                                new_batch.non_tensor_batch["data_source"] = np.array(
                                    ["math_dapo"] * n_items, dtype=object)

                            # If you attach extra info from reward_fn, make it indexable *before* updating:
                            try:
                                reward_result = self.reward_fn(
                                    new_batch, return_dict=True)
                                reward_tensor = reward_result["reward_tensor"]
                                reward_extra_infos_dict = reward_result.get(
                                    "reward_extra_info", {})
                            except Exception as e:
                                logger.warning(
                                    f"Error in reward_fn (using fallback): {e}")
                                reward_tensor = self.reward_fn(new_batch)
                                reward_extra_infos_dict = {}

                            # IMPORTANT: make each extra-info field indexable and aligned to batch size
                            n_items = len(new_batch.batch["input_ids"])
                            for k, v in reward_extra_infos_dict.items():
                                arr = _to_indexable_array(v, n_items)
                                # Force 1-D (B,) object array if someone handed us a (B, L) nd array
                                if isinstance(arr, np.ndarray) and arr.ndim != 1:
                                    arr = np.array(
                                        [arr[i] for i in range(n_items)], dtype=object)
                                new_batch.non_tensor_batch[k] = arr

                            # One more safety pass on the whole dict (catches anything else)
                            _sanitize_non_tensor_batch(new_batch)
                            if batch is not None:
                                _sanitize_non_tensor_batch(batch)

                            # Continue as before
                            new_batch.batch["token_level_scores"] = reward_tensor

                            if self.config.algorithm.use_kl_in_reward:
                                new_batch, kl_metrics = apply_kl_penalty(
                                    new_batch,
                                    kl_ctrl=self.kl_ctrl_in_reward,
                                    kl_penalty=self.config.algorithm.kl_penalty
                                )
                                metrics.update(kl_metrics)
                            else:
                                new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                            # ---- attach actual per-UID reward and update sampled records/archive ----
                            try:
                                # per-trajectory scalar
                                traj_rewards = new_batch.batch["token_level_rewards"].sum(
                                    dim=-1).detach().cpu().numpy()
                                uids = list(new_batch.non_tensor_batch["uid"])
                                uid2rewards = defaultdict(list)
                                for r, u in zip(traj_rewards, uids):
                                    uid2rewards[u].append(float(r))

                                # Preserve base order of unique UIDs
                                base_uid_order = []
                                seen = set()
                                for u in uids:
                                    if u not in seen:
                                        base_uid_order.append(u)
                                        seen.add(u)

                                # Map sampled records -> their avg reward (may skip if a UID got filtered out upstream)
                                for i, rec in enumerate(sampled_records_for_this_step):
                                    if i >= len(base_uid_order):
                                        break
                                    u = base_uid_order[i]
                                    if u not in uid2rewards:
                                        continue  # e.g., all repeats filtered out
                                    avg_r = float(np.mean(uid2rewards[u]))
                                    # write back into the record
                                    rec.reward = avg_r
                                    rec.est_reward = avg_r
                                    # replace its meta snapshot
                                    rec.meta = {**(rec.meta or {}), "origin": (rec.meta or {}).get(
                                        "origin", "seed"), "trained_round": self.global_steps}
                                    # archive / upsert
                                    self._trained_archive[rec.record_id] = self.query_pool._copy_record_unlocked(
                                        rec)

                                # (Optional) also make this visible in the batch for downstream logging
                                # broadcast per-uid avg reward over repeats
                                per_traj_avg = []
                                for u in uids:
                                    vals = uid2rewards.get(u, [])
                                    per_traj_avg.append(
                                        float(np.mean(vals)) if vals else np.nan)
                                new_batch.non_tensor_batch["actual_avg_reward"] = np.asarray(
                                    per_traj_avg, dtype=float)

                                # Build base-reward overrides for augmentation by normalized text hash
                                base_reward_overrides_hash = {}
                                for i, rec in enumerate(sampled_records_for_this_step):
                                    if i >= len(base_uid_order):
                                        break
                                    u = base_uid_order[i]
                                    if u not in uid2rewards:
                                        continue
                                    avg_r = float(np.mean(uid2rewards[u]))
                                    key = self._text_hash(
                                        rec.original_text or self._decode_tokens_to_text(rec.raw_prompt_data))
                                    base_reward_overrides_hash[key] = avg_r

                            except Exception as e:
                                logger.debug(
                                    f"attach-actual-reward failed: {e}")
                                base_reward_overrides_hash = {}

                        # Step 4: policy-driven augmentation (THROTTLED)
                        if do_augment:
                            try:
                                self.query_pool.set_max_size(
                                    int(getattr(self.config.dynamic_data, "max_pool_size", 500000)))
                                remain = self.query_pool.capacity_remaining()

                                want_per_prompt = int(
                                    aug_cfg.get("num_per_prompt", 1))
                                # Number of ORIGINAL prompts in this step (before rollout repeats).
                                # We already computed these above as base_bsz; recompute defensively if needed.
                                try:
                                    num_prompts = int(base_bsz)
                                except NameError:
                                    try:
                                        n_rep = int(
                                            self.config.actor_rollout_ref.rollout.n)
                                        gen_bsz = int(
                                            gen_batch.batch["input_ids"].size(0))
                                        num_prompts = max(
                                            1, gen_bsz // max(1, n_rep))
                                    except Exception:
                                        # Fallback: approximate by unique raw prompts
                                        rp = list(gen_batch.non_tensor_batch.get(
                                            "raw_prompt_data", []))
                                        num_prompts = max(
                                            1, len({str(x) for x in rp}))
                                required = want_per_prompt * num_prompts

                                if want_per_prompt > 0 and num_prompts > 0 and remain >= required:
                                    # NEW: pass base reward overrides (actual per-prompt reward)
                                    aug_cfg_this = dict(aug_cfg or {})
                                    aug_cfg_this["base_reward_overrides_hash"] = base_reward_overrides_hash
                                    aug_proto = self.generate_augmented_queries(
                                        source_batch=gen_batch,
                                        num_per_prompt=want_per_prompt,
                                        aug_cfg=aug_cfg_this,
                                    )
                                    if self.teacher_annotator is not None and len(aug_proto.batch["input_ids"]) > 0:
                                        if not self.teacher_annotator.enqueue_aug(aug_proto):
                                            metrics["augmentation/queue_full_events"] = metrics.get(
                                                "augmentation/queue_full_events", 0) + 1
                                else:
                                    # Log why we skipped (helps debugging)
                                    metrics["augmentation/skipped_due_to_capacity"] = metrics.get(
                                        "augmentation/skipped_due_to_capacity", 0) + 1
                                    metrics["augmentation/required_capacity"] = required
                                    metrics["augmentation/capacity_remaining"] = remain
                                    metrics["augmentation/num_prompts"] = num_prompts
                                    metrics["augmentation/num_per_prompt"] = want_per_prompt

                            except Exception as e:
                                logger.error(
                                    f"[dynamic] augmentation error: {e}", exc_info=True)

                        # (DAPO group filtering) - keep original logic
                        if not self.config.algorithm.filter_groups.enable:
                            batch = new_batch
                        else:
                            metric_name = self.config.algorithm.filter_groups.metric
                            if metric_name == "seq_final_reward":
                                new_batch.non_tensor_batch["seq_final_reward"] = (
                                    new_batch.batch["token_level_rewards"].sum(
                                        dim=-1).numpy()
                                )
                            elif metric_name == "seq_reward":
                                new_batch.non_tensor_batch["seq_reward"] = (
                                    new_batch.batch["token_level_scores"].sum(
                                        dim=-1).numpy()
                                )

                            prompt_uid2metric_vals = defaultdict(list)
                            for uid, metric_val in zip(
                                new_batch.non_tensor_batch["uid"],
                                new_batch.non_tensor_batch[metric_name]
                            ):
                                prompt_uid2metric_vals[uid].append(metric_val)

                            prompt_uid2metric_std = {
                                uid: np.std(vals)
                                for uid, vals in prompt_uid2metric_vals.items()
                            }

                            kept_prompt_uids = [
                                uid for uid, std in prompt_uid2metric_std.items()
                                if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                            ]
                            num_prompt_in_batch += len(kept_prompt_uids)

                            kept_traj_idxs = [
                                idx for idx, traj_uid in enumerate(new_batch.non_tensor_batch["uid"])
                                if traj_uid in kept_prompt_uids
                            ]

                            new_batch = new_batch[kept_traj_idxs]
                            if batch is None:
                                batch = new_batch
                            else:
                                a, b = self._pad_for_concat([batch, new_batch])
                                # ensure both sides have identical tensor keys
                                ka, kb = set(a.batch.keys()), set(
                                    b.batch.keys())
                                extra_a, extra_b = ka - kb, kb - ka
                                # simplest safe policy: drop extras (or fill with zeros if you prefer)
                                for k in list(extra_a):
                                    a.batch.pop(k)
                                for k in list(extra_b):
                                    b.batch.pop(k)
                                batch = DataProto.concat([a, b])

                            prompt_bsz = self.config.data.train_batch_size
                            if num_prompt_in_batch < prompt_bsz:
                                logger.info(
                                    f"{num_prompt_in_batch=} < {prompt_bsz=}")
                                max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    logger.info(
                                        f"{num_gen_batches=}. Keep generating...")
                                    continue
                                else:
                                    raise ValueError(
                                        f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                        + " Generated too many. Please check if your data are too difficult."
                                        + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                    )
                            else:
                                traj_bsz = self.config.data.train_batch_size * \
                                    self.config.actor_rollout_ref.rollout.n
                                batch = batch[:traj_bsz]

                        # Step 7: PPO updating
                        batch.batch["response_mask"] = compute_response_mask(
                            batch)

                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        batch.meta_info["global_token_num"] = torch.sum(
                            batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        with marked_timer("old_log_prob", timing_raw, "blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(
                                batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]

                            # Align lengths: entropys is for next-token preds (T-1), mask is over tokens (T)
                            if response_masks.size(-1) != entropys.size(-1):
                                if response_masks.size(-1) == entropys.size(-1) + 1:
                                    # Drop the first token so the mask covers target positions (tokens 1..T-1)
                                    response_masks = response_masks[..., 1:]
                                else:
                                    # Defensive fallback: trim both to the common length
                                    T = min(response_masks.size(-1),
                                            entropys.size(-1))
                                    response_masks = response_masks[..., :T]
                                    entropys = entropys[..., :T]

                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=loss_agg_mode,
                            )

                            metrics.update(
                                {"actor/entropy": entropy_agg.detach().item()})
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                            # === Align per-token time dimensions to next-token length (T-1) ===
                            def _infer_pred_len(b):
                                td = b.batch  # a TensorDict
                                # Check a wide set of possible per-token tensors and return the first match
                                candidates = (
                                    "log_probs", "old_log_probs", "ref_log_probs",    # snake_case
                                    "logprobs", "old_logprobs", "ref_logprobs",       # camel-less
                                    "entropys", "approx_kls",
                                    "token_level_rewards", "token_level_scores",
                                )
                                for k in candidates:
                                    if k in td.keys():
                                        t = td[k]
                                        if isinstance(t, torch.Tensor) and t.dim() >= 2:
                                            return t.size(-1)

                                # Fallback: next-token length from attention mask
                                if "attention_mask" in td:
                                    am = td["attention_mask"]
                                    return max(am.size(-1) - 1, 0)
                                return 0  # Ultimate fallback

                            # Before update_actor: align inputs to pred_len
                            pred_len = _infer_pred_len(batch)

                            def _align_lastdim(x, L):
                                if x.size(-1) == L:
                                    return x
                                if x.size(-1) == L + 1:
                                    return x[..., 1:]
                                return x[..., :min(x.size(-1), L)]

                            for k in (
                                "response_mask",
                                "token_level_scores",
                                "token_level_rewards",
                                "log_probs", "old_log_probs", "ref_log_probs",     # snake_case
                                "logprobs", "old_logprobs", "ref_logprobs",        # alt names
                                "approx_kls", "values", "old_values",
                            ):
                                if k in batch.batch.keys():
                                    t = batch.batch[k]
                                    if isinstance(t, torch.Tensor) and t.dim() >= 2:
                                        batch.batch[k] = _align_lastdim(
                                            t, pred_len)

                        if self.use_reference_policy:
                            with marked_timer("ref", timing_raw, "olive"):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                    batch)
                                batch = batch.union(ref_log_prob)

                        if self.use_critic:
                            with marked_timer("values", timing_raw, "cyan"):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, "brown"):
                            norm_adv_by_std_in_grpo = self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True)
                            batch = compute_advantage(
                                batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            )

                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, "pink"):
                                critic_output = self.critic_wg.update_critic(
                                    batch)
                            metrics.update(reduce_metrics(
                                critic_output.meta_info["metrics"]))

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with marked_timer("update_actor", timing_raw, "red"):
                                # Align response_mask (T) to next-token prediction length (T-1)
                                mask = batch.batch["response_mask"]
                                seq_len = batch.batch["attention_mask"].size(
                                    -1)
                                # next-token targets
                                pred_len = max(seq_len - 1, 0)

                                if mask.size(-1) != pred_len:
                                    if mask.size(-1) == pred_len + 1:
                                        # Drop the first position so mask covers targets 1..T-1
                                        mask = mask[..., 1:]
                                    else:
                                        # Defensive fallback for any other mismatch
                                        T = min(mask.size(-1), pred_len)
                                        mask = mask[..., :T]

                                batch.batch["response_mask"] = mask

                                actor_output = self.actor_rollout_wg.update_actor(
                                    batch)
                            metrics.update(reduce_metrics(
                                actor_output.meta_info["metrics"]))

                    # Validate / save
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                            (is_last_step or (self.global_steps % self.config.trainer.test_freq) == 0):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    save_freq = int(self.config.trainer.save_freq)
                    if self.config.trainer.save_freq > 0 and (is_last_step or (self.global_steps % self.config.trainer.save_freq) == 0):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()

                    # Collect all metrics
                    metrics.update(compute_data_metrics(
                        batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(
                        batch=batch, timing_raw=timing_raw))
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    ))
                    timing_raw = defaultdict(float)

                    # Add custom metrics
                    metrics["train/num_gen_batches"] = num_gen_batches
                    metrics["dynamic/pool_size"] = self.query_pool.size() if dyn_en else 0
                    metrics["dynamic/capacity_remaining"] = self.query_pool.capacity_remaining() if dyn_en else 0

                    # Add pool metrics
                    if dyn_en:
                        pool_metrics = self.query_pool.get_metrics()
                        for k, v in pool_metrics.items():
                            metrics[f"pool/{k}"] = v

                    # Add augmentation metrics
                    for k, v in self._augmentation_metrics.items():
                        metrics[f"augmentation/{k}"] = v

                    # Add teacher annotator metrics if available
                    if self.teacher_annotator:
                        teacher_metrics = self.teacher_annotator.get_metrics()
                        for k, v in teacher_metrics.items():
                            metrics[f"teacher/{k}"] = v

                    metrics["delayed/released_this_step"] = released_now
                    metrics.update(self._delayed_buffer_metrics())

                    # Compute ratio of augmented items in the training batch
                    try:
                        is_aug = list(
                            batch.non_tensor_batch.get("is_augmented", []))
                        if is_aug:
                            aug_cnt = int(np.sum(is_aug))
                            total = len(is_aug)
                        else:
                            origins = list(
                                batch.non_tensor_batch.get("origin", []))
                            if origins:
                                aug_cnt = sum(
                                    1 for x in origins if str(x) == "augmented")
                                total = len(origins)
                            else:
                                ds = list(batch.non_tensor_batch.get(
                                    "data_source", []))
                                aug_cnt = sum(
                                    1 for x in ds if str(x) == "math_dapo")
                                total = len(ds)
                        metrics["batch/augmented_ratio"] = aug_cnt / \
                            max(1, total)
                        metrics["batch/augmented_count"] = aug_cnt
                        metrics["batch/seed_count"] = max(0, total - aug_cnt)
                    except Exception as e:
                        logger.debug(f"augmented ratio calc failed: {e}")

                    # also log distribution stats
                    try:
                        seq_rewards = batch.batch["token_level_rewards"].sum(
                            dim=-1).detach().cpu()
                        metrics["batch/reward_mean"] = float(
                            seq_rewards.mean())
                        metrics["batch/reward_std"] = float(
                            seq_rewards.std(unbiased=False))
                    except Exception:
                        pass

                    # ---- Per-origin reward stats (augmented vs original) ----
                    try:
                        seq_rewards = batch.batch["token_level_rewards"].sum(
                            dim=-1).detach().cpu()
                        if seq_rewards.numel() == 0:
                            metrics["augmented/reward/count"] = 0
                            metrics["original/reward/count"] = 0
                        else:
                            is_aug_list = list(
                                batch.non_tensor_batch.get("is_augmented", []))
                            if not is_aug_list:
                                origins = list(
                                    batch.non_tensor_batch.get("origin", []))
                                if origins:
                                    is_aug_list = [
                                        str(x) == "augmented" for x in origins]
                                else:
                                    ds = list(batch.non_tensor_batch.get(
                                        "data_source", []))
                                    is_aug_list = [
                                        str(x) == "math_dapo" for x in ds]

                            L = seq_rewards.numel()
                            if len(is_aug_list) < L:
                                # default to “original”
                                is_aug_list += [False] * (L - len(is_aug_list))
                            elif len(is_aug_list) > L:
                                is_aug_list = is_aug_list[:L]
                            mask = torch.tensor(is_aug_list, dtype=torch.bool)

                            aug, orig = seq_rewards[mask], seq_rewards[~mask]
                            metrics["augmented/reward/count"] = int(
                                aug.numel())
                            metrics["original/reward/count"] = int(
                                orig.numel())
                            if aug.numel():
                                metrics["augmented/reward/mean"] = float(
                                    aug.mean())
                                metrics["augmented/reward/min"] = float(
                                    aug.min())
                                metrics["augmented/reward/max"] = float(
                                    aug.max())
                            if orig.numel():
                                metrics["original/reward/mean"] = float(
                                    orig.mean())
                                metrics["original/reward/min"] = float(
                                    orig.min())
                                metrics["original/reward/max"] = float(
                                    orig.max())
                    except Exception as e:
                        logger.debug(f"Per-origin reward logging failed: {e}")

                    batch = None
                    num_prompt_in_batch = 0
                    num_gen_batches = 0

                    logger_instance.log(data=metrics, step=self.global_steps)

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        self._cleanup_teacher_annotator()
                        return

                    progress_bar.update(1)
                    self.global_steps += 1
                    self.gen_steps += 1

            # Final checkpoint
            checkpoint_dir = os.path.join(
                self.config.trainer.default_local_dir,
                f"global_step_{self.global_steps}"
            )
            if self.global_steps > 0 and not os.path.exists(checkpoint_dir):
                timing_raw = defaultdict(float)
                with marked_timer("save_checkpoint", timing_raw, "green"):
                    self._save_checkpoint()
                metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
                logger_instance.log(data=metrics, step=self.global_steps)

        except Exception as e:
            logger.error(f"Training loop error: {e}", exc_info=True)
            raise
        finally:
            self._cleanup_teacher_annotator()

    def _cleanup_teacher_annotator(self):
        """Properly cleanup teacher annotator thread and save final logs."""
        teacher = getattr(self, "teacher_annotator", None)
        if teacher is not None:
            logger.info("Shutting down teacher annotator...")
            try:
                teacher.shutdown()
            except Exception as e:
                logger.warning(
                    f"Error during teacher annotator shutdown(): {e}")
            try:
                teacher.join(timeout=10.0)
                if teacher.is_alive():
                    logger.warning(
                        "Teacher annotator thread did not terminate cleanly")
            except Exception as e:
                logger.warning(f"Error joining teacher annotator thread: {e}")
            self.teacher_annotator = None
            logger.info("Teacher annotator shutdown complete")

        # Final save of augmentation logs (guarded)
        aug_logger = getattr(self, "augmentation_logger", None)
        if aug_logger is not None:
            try:
                aug_logger.flush_all()
                logger.info("All augmentation logs saved")
            except Exception as e:
                logger.warning(f"Error flushing augmentation logs: {e}")
