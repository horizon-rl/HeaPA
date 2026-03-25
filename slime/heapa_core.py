"""
heapa_core.py - Core data structures for HeaPA (Heap-based on-Policy Query Augmentation).

Extracted from dapo_ray_trainer_HeaPA.py, stripped of VERL dependencies so they can
be used directly in the slime training framework.
"""
from __future__ import annotations

import heapq
import itertools
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

logger = logging.getLogger(__name__)


# ==============================
# QueryRecord
# ==============================

@dataclass
class QueryRecord:
    """Lightweight record kept in the query pool."""
    raw_prompt_data: np.ndarray          # 1-D int array of token IDs
    input_ids: Optional[Any] = None      # cached tensor (optional)
    attention_mask: Optional[Any] = None
    position_ids: Optional[Any] = None
    gt: Optional[object] = None          # ground truth answer
    reward: Optional[float] = None       # actual reward (set after rollout)
    est_reward: Optional[float] = None   # estimated reward for augmented queries
    meta: dict = field(default_factory=dict)

    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_text: Optional[str] = None
    augmented_text: Optional[str] = None
    teacher_response: Optional[str] = None
    creation_time: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
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
# ThreadSafeQueryPool
# ==============================

class ThreadSafeQueryPool:
    """
    Two-partition heap-based query pool with reward-aware sampling.

    low_heap  : max-heap on reward (stores worst/hardest queries for medium sampling).
    high_heap : min/max twin heap with lazy deletion (stores best/easiest queries).
    cold_queue: FIFO for unscored items (reward=None), processed first to get them scored.

    Sampling policy:
      - sample_batch(k): pull from cold queue first, then medium partition.
      - "medium" = low_heap items sampled uniformly.
      - optional "mixed" mode = half easy (from high_heap_max) + half medium.
    """

    def __init__(
        self,
        max_size: int = 30000,
        low_fraction: float = 0.5,
        rng: Optional[np.random.Generator] = None,
        cleanup_frequency: int = 1000,
        mixed_easy_medium: bool = False,
    ):
        self._lock = threading.RLock()
        self._max_size = max(1, int(max_size))
        self._low_fraction = float(np.clip(low_fraction, 0.05, 0.95))
        self._cleanup_frequency = cleanup_frequency
        self._operations_count = 0
        self._mixed_easy_medium = bool(mixed_easy_medium)

        # Heap tuples: (key, seq, QueryRecord)
        self._low_heap: List[Tuple[float, int, QueryRecord]] = []
        self._high_heap_min: List[Tuple[float, int, QueryRecord]] = []
        self._high_heap_max: List[Tuple[float, int, QueryRecord]] = []
        self._active_high: set = set()

        # Cold queue: unscored items (reward=None)
        self._cold_queue: deque = deque()

        self._seq = itertools.count()
        self._rng = rng if rng is not None else np.random.default_rng()

        self._total_added = 0
        self._total_sampled = 0
        self._total_evicted = 0

    def set_mixed_easy_medium(self, enabled: bool) -> None:
        with self._lock:
            self._mixed_easy_medium = bool(enabled)

    def set_max_size(self, max_size: int):
        with self._lock:
            old_size = self._max_size
            self._max_size = max(1, int(max_size))
            if self._max_size < old_size:
                self._evict_to_capacity_unlocked()

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
                "cold_queue_size": len(self._cold_queue),
                "capacity_remaining": self.capacity_remaining(),
                "total_added": self._total_added,
                "total_sampled": self._total_sampled,
                "total_evicted": self._total_evicted,
            }

    def initialize_uniform(self, seed_items: List[QueryRecord]):
        """Add seed items (typically reward=None) to the pool."""
        self.add_many(seed_items)
        logger.info(f"Pool initialized with {len(seed_items)} seed items")

    # ---------- insertion ----------

    def add_many(self, items: List[QueryRecord]):
        """Add multiple records; evict if over capacity."""
        if not items:
            return
        with self._lock:
            for it in items:
                # Validate: raw_prompt_data must be 1-D integer ndarray
                data = it.raw_prompt_data
                if not isinstance(data, np.ndarray) or data.ndim != 1 or not np.issubdtype(data.dtype, np.integer):
                    continue
                if data.size == 0:
                    continue
                self._insert_unlocked(it)
                self._total_added += 1
            self._evict_to_capacity_unlocked()

    def _insert_unlocked(self, it: QueryRecord):
        """Insert one record (no lock, caller must hold it)."""
        if it.reward is None:
            self._cold_queue.append(it)
        else:
            self._insert_scored_unlocked(it)

    def _insert_scored_unlocked(self, it: QueryRecord):
        """Insert a scored record into low or high partition."""
        low_cap = int(self._max_size * self._low_fraction)
        if len(self._low_heap) < low_cap:
            self._push_low_unlocked(it)
        else:
            # Decide: does it belong in high (better than worst low)?
            worst_low_reward = -self._low_heap[0][0] if self._low_heap else float("-inf")
            if it.reward >= worst_low_reward:
                self._push_high_unlocked(it)
            else:
                self._push_low_unlocked(it)

    def _evict_to_capacity_unlocked(self):
        """Remove items until size <= max_size."""
        while len(self._low_heap) + self._high_size_unlocked() + len(self._cold_queue) > self._max_size:
            # Evict from high first (easier items)
            if self._high_size_unlocked() > 0:
                self._remove_high_heap_min_unlocked()
            elif self._low_heap:
                self._remove_low_heap_min_unlocked()
            elif self._cold_queue:
                self._cold_queue.popleft()
                self._total_evicted += 1
            else:
                break

    def _remove_high_heap_min_unlocked(self) -> bool:
        self._clean_high_min_top_unlocked()
        while self._high_heap_min:
            r, seq, _ = heapq.heappop(self._high_heap_min)
            if seq in self._active_high:
                self._active_high.remove(seq)
                self._total_evicted += 1
                return True
        return False

    def _remove_low_heap_min_unlocked(self) -> bool:
        if not self._low_heap:
            return False
        worst_i = max(range(len(self._low_heap)), key=lambda i: self._low_heap[i][0])
        self._low_heap[worst_i] = self._low_heap[-1]
        self._low_heap.pop()
        heapq.heapify(self._low_heap)
        self._total_evicted += 1
        return True

    # ---------- sampling ----------

    def sample_batch(self, k: int) -> List[QueryRecord]:
        """Sample k records for rollout (no removal from pool, returns copies)."""
        if k <= 0:
            return []
        with self._lock:
            n_total = len(self._low_heap) + self._high_size_unlocked() + len(self._cold_queue)
            if n_total == 0:
                return []

            actual_k = min(k, n_total)

            # Cold queue first (get unscored items scored)
            cold_take = min(actual_k, len(self._cold_queue))
            cold = [self._cold_queue.popleft() for _ in range(cold_take)]

            need = actual_k - cold_take
            if need <= 0:
                chosen = cold
            else:
                if self._mixed_easy_medium:
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
                    self._rebalance_low_high_unlocked()
                else:
                    medium = self._sample_medium_k_unlocked(need)
                    chosen = cold + medium

            self._total_sampled += len(chosen)
            return [self._copy_record_unlocked(r) for r in chosen]

    def _sample_medium_k_unlocked(self, k: int) -> List[QueryRecord]:
        """Sample k items from low_heap (uniform random, without removal)."""
        if k <= 0 or not self._low_heap:
            return []
        k = min(k, len(self._low_heap))
        indices = self._rng.choice(len(self._low_heap), size=k, replace=False)
        return [self._low_heap[i][2] for i in indices]

    def _pop_high_max_unlocked(self):
        """Pop the highest-reward item from the high partition."""
        self._clean_high_max_top_unlocked()
        while self._high_heap_max:
            neg_r, seq, it = heapq.heappop(self._high_heap_max)
            if seq in self._active_high:
                self._active_high.remove(seq)
                return (-float(neg_r), seq, it)
        return None

    def _rebalance_low_high_unlocked(self, local_only: bool = False):
        """Move items between partitions to maintain the low_fraction ratio."""
        low_cap = int(self._max_size * self._low_fraction)
        # Move overflow from low → high
        while len(self._low_heap) > low_cap and self._low_heap:
            worst_i = max(range(len(self._low_heap)), key=lambda i: -self._low_heap[i][0])
            _, _, it = self._low_heap[worst_i]
            self._low_heap[worst_i] = self._low_heap[-1]
            self._low_heap.pop()
            heapq.heapify(self._low_heap)
            self._push_high_unlocked(it)

    def _push_low_unlocked(self, it: QueryRecord):
        heapq.heappush(self._low_heap, (-float(it.reward), next(self._seq), it))

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

    def _copy_record_unlocked(self, r: QueryRecord) -> QueryRecord:
        """Shallow copy of a record (raw_prompt_data is shared, which is fine for read-only use)."""
        return QueryRecord(
            raw_prompt_data=r.raw_prompt_data,
            input_ids=r.input_ids,
            attention_mask=r.attention_mask,
            position_ids=r.position_ids,
            gt=r.gt,
            reward=r.reward,
            est_reward=r.est_reward,
            meta=dict(r.meta) if r.meta else {},
            record_id=r.record_id,
            original_text=r.original_text,
            augmented_text=r.augmented_text,
            teacher_response=r.teacher_response,
            creation_time=r.creation_time,
        )

    def update_reward(self, record_id: str, reward: float) -> bool:
        """Update the reward of an existing record in the pool by record_id.

        The record stays in the same partition (low/high). The reward value is
        updated in-place; the heap property may drift slightly but that's acceptable
        since we periodically re-insert via add_many.

        Returns True if found, False otherwise.
        """
        with self._lock:
            # Search cold queue
            for rec in self._cold_queue:
                if rec.record_id == record_id:
                    rec.reward = reward
                    # Move from cold to scored partition
                    self._cold_queue.remove(rec)
                    self._insert_scored_unlocked(rec)
                    return True
            # Search low_heap
            for _, _, rec in self._low_heap:
                if rec.record_id == record_id:
                    rec.reward = reward
                    return True
            # Search high (active)
            for _, seq, rec in self._high_heap_min:
                if seq in self._active_high and rec.record_id == record_id:
                    rec.reward = reward
                    return True
        return False

    def get_all_records(self) -> List[QueryRecord]:
        """Return a snapshot of all records (for save/load)."""
        with self._lock:
            records = []
            records.extend(rec for _, _, rec in self._low_heap)
            records.extend(rec for _, seq, rec in self._high_heap_min if seq in self._active_high)
            records.extend(self._cold_queue)
            return records

    def restore_from_records(self, records: List[QueryRecord]):
        """Re-populate pool from a saved list of records (e.g. after load)."""
        with self._lock:
            self._low_heap.clear()
            self._high_heap_min.clear()
            self._high_heap_max.clear()
            self._active_high.clear()
            self._cold_queue.clear()
            for rec in records:
                self._insert_unlocked(rec)
