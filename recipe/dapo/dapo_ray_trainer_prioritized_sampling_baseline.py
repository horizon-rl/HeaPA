from __future__ import annotations

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
from typing import Dict, List, Optional, Tuple, Any, Union
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

# ==============================
# Serialization Helpers for Complex Data Structures
# ==============================

def _serialize_numpy_object_array(arr: np.ndarray) -> Dict[str, Any]:
    """
    Safely serialize numpy object arrays that may contain nested structures.
    Returns a dict that can be passed to torch.save.
    """
    if not isinstance(arr, np.ndarray):
        # Not an array - return as-is
        return {'type': 'raw', 'data': arr}
    
    if arr.dtype != object:
        # Regular numeric array - can serialize directly
        return {'type': 'ndarray', 'data': arr}
    
    # Object array - need to extract elements carefully
    try:
        flat = arr.flatten()
        serialized_elements = []
        
        for elem in flat:
            if elem is None:
                serialized_elements.append({'type': 'none'})
            elif isinstance(elem, (str, int, float, bool)):
                serialized_elements.append({'type': 'primitive', 'data': elem})
            elif isinstance(elem, np.ndarray):
                # Nested array - recurse
                if elem.ndim == 0:
                    # 0-d array - extract scalar
                    serialized_elements.append({'type': '0d_array', 'data': elem.item(), 'dtype': str(elem.dtype)})
                else:
                    serialized_elements.append({'type': 'nested_array', 'data': elem.copy()})
            elif isinstance(elem, dict):
                serialized_elements.append({'type': 'dict', 'data': elem})
            elif isinstance(elem, (list, tuple)):
                serialized_elements.append({'type': 'sequence', 'data': list(elem), 'original_type': type(elem).__name__})
            else:
                # Unknown type - try to pickle it
                try:
                    import pickle
                    serialized_elements.append({'type': 'pickled', 'data': pickle.dumps(elem)})
                except:
                    # Last resort - store string representation
                    serialized_elements.append({'type': 'string_repr', 'data': str(elem)})
        
        return {
            'type': 'object_array',
            'shape': arr.shape,
            'elements': serialized_elements
        }
    except Exception as e:
        logger.warning(f"Failed to serialize object array: {e}, storing as string")
        return {'type': 'fallback', 'data': str(arr)}

def _deserialize_numpy_object_array(serialized: Dict[str, Any]) -> Any:
    """Reconstruct numpy object arrays from serialized form."""
    arr_type = serialized['type']
    
    if arr_type == 'raw':
        return serialized['data']
    elif arr_type == 'ndarray':
        return serialized['data']
    elif arr_type == 'fallback':
        logger.warning("Loading fallback string representation")
        return serialized['data']
    elif arr_type == 'object_array':
        shape = serialized['shape']
        elements = serialized['elements']
        
        reconstructed = []
        for elem_dict in elements:
            elem_type = elem_dict['type']
            if elem_type == 'none':
                reconstructed.append(None)
            elif elem_type == 'primitive':
                reconstructed.append(elem_dict['data'])
            elif elem_type == '0d_array':
                # Reconstruct 0-d array with proper dtype
                dtype = np.dtype(elem_dict['dtype']) if 'dtype' in elem_dict else None
                reconstructed.append(np.array(elem_dict['data'], dtype=dtype))
            elif elem_type == 'nested_array':
                reconstructed.append(elem_dict['data'])
            elif elem_type == 'dict':
                reconstructed.append(elem_dict['data'])
            elif elem_type == 'sequence':
                data = elem_dict['data']
                orig_type = elem_dict.get('original_type', 'list')
                reconstructed.append(tuple(data) if orig_type == 'tuple' else data)
            elif elem_type == 'pickled':
                import pickle
                reconstructed.append(pickle.loads(elem_dict['data']))
            elif elem_type == 'string_repr':
                reconstructed.append(elem_dict['data'])
        
        arr = np.array(reconstructed, dtype=object)
        return arr.reshape(shape)
    
    return None

def _serialize_tensordict(td) -> Dict[str, Any]:
    """
    Serialize TensorDict to a plain dict of CPU tensors.
    Handles both tensordict.TensorDict and torchrl.data.TensorDict.
    """
    if td is None:
        return None
    
    # Extract all tensors and move to CPU
    plain_dict = {}
    for key in td.keys():
        value = td[key]
        if isinstance(value, torch.Tensor):
            plain_dict[key] = value.detach().cpu()
        else:
            # Nested TensorDict or other structure
            plain_dict[key] = value
    
    return {
        'batch_size': list(td.batch_size) if hasattr(td, 'batch_size') else None,
        'data': plain_dict
    }

def _deserialize_tensordict(serialized: Dict[str, Any]):
    """Reconstruct TensorDict from serialized form."""
    if serialized is None:
        return None
    
    try:
        from tensordict import TensorDict
    except:
        from torchrl.data import TensorDict
    
    data = serialized['data']
    batch_size = serialized.get('batch_size')
    
    if batch_size:
        return TensorDict(data, batch_size=batch_size)
    else:
        # Infer batch size from first tensor
        for v in data.values():
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                return TensorDict(data, batch_size=[v.size(0)])
        return TensorDict(data, batch_size=[0])

def _serialize_dataproto(dp: DataProto) -> Dict[str, Any]:
    """Serialize a DataProto object."""
    if dp is None:
        return None
    
    # Serialize the TensorDict batch
    batch_serialized = _serialize_tensordict(dp.batch)
    
    # Serialize non_tensor_batch (dict of numpy arrays/objects)
    non_tensor_serialized = {}
    for key, value in dp.non_tensor_batch.items():
        non_tensor_serialized[key] = _serialize_numpy_object_array(value)
    
    return {
        'batch': batch_serialized,
        'non_tensor_batch': non_tensor_serialized,
        'meta_info': dp.meta_info
    }

def _deserialize_dataproto(serialized: Dict[str, Any]) -> DataProto:
    """Reconstruct a DataProto object."""
    if serialized is None:
        return None
    
    # Deserialize TensorDict
    batch = _deserialize_tensordict(serialized['batch'])
    
    # Deserialize non_tensor_batch
    non_tensor_batch = {}
    for key, value_serialized in serialized['non_tensor_batch'].items():
        non_tensor_batch[key] = _deserialize_numpy_object_array(value_serialized)
    
    meta_info = serialized.get('meta_info', {})
    
    return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

def _serialize_query_record(rec) -> Dict[str, Any]:
    """Serialize a QueryRecord with all its fields."""
    return {
        'raw_prompt_data': _serialize_numpy_object_array(rec.raw_prompt_data),
        'input_ids': rec.input_ids.detach().cpu() if rec.input_ids is not None else None,
        'attention_mask': rec.attention_mask.detach().cpu() if rec.attention_mask is not None else None,
        'position_ids': rec.position_ids.detach().cpu() if rec.position_ids is not None else None,
        'gt': rec.gt,  # Can be any type
        'reward': float(rec.reward) if rec.reward is not None and np.isfinite(rec.reward) else None,
        'est_reward': float(rec.est_reward) if rec.est_reward is not None and np.isfinite(rec.est_reward) else None,
        'meta': dict(rec.meta) if rec.meta else {},
        'record_id': rec.record_id,
        'original_text': rec.original_text,
        'augmented_text': rec.augmented_text,
        'teacher_response': rec.teacher_response,
        'creation_time': rec.creation_time,
    }

def _deserialize_query_record(serialized: Dict[str, Any]) -> QueryRecord:
    """Reconstruct a QueryRecord from serialized form."""
    return QueryRecord(
        raw_prompt_data=_deserialize_numpy_object_array(serialized['raw_prompt_data']),
        input_ids=serialized['input_ids'],
        attention_mask=serialized['attention_mask'],
        position_ids=serialized['position_ids'],
        gt=serialized['gt'],
        reward=serialized['reward'],
        est_reward=serialized['est_reward'],
        meta=serialized['meta'],
        record_id=serialized['record_id'],
        original_text=serialized['original_text'],
        augmented_text=serialized['augmented_text'],
        teacher_response=serialized['teacher_response'],
        creation_time=serialized['creation_time'],
    )
    
# ==============================
# TensorDict utilities
# ==============================

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
    
    ALSO FIXES CORRUPTION: 0-d arrays, wrong dtypes
    """
    import numpy as _np

    # FIX: Handle 0-d arrays (corruption source)
    if isinstance(v, _np.ndarray) and v.ndim == 0:
        scalar = v.item()  # Extract value
        
        # If it's text (corruption), try to recover by tokenizing
        if isinstance(scalar, (str, bytes)):
            print(f"Found 0-d text array in sanitization: '{str(scalar)[:50]}...'. Attempting recovery.")
            # Don't broadcast corruption - this needs manual fix
            # For now, mark as invalid
            return _np.array([None] * n, dtype=object)
        
        # Numeric scalar - safe to broadcast
        return _np.repeat(scalar, n)

    # Already a numpy array of correct length
    if isinstance(v, _np.ndarray):
        # FIX: Ensure 1-D arrays are proper, not nested 0-d
        if v.ndim == 1 and len(v) == n:
            # Check if elements are themselves 0-d arrays
            try:
                if v.dtype == object and len(v) > 0:
                    first = v[0]
                    if isinstance(first, _np.ndarray) and first.ndim == 0:
                        print("Fixing nested 0-d arrays in sanitization")
                        # Unwrap each element
                        return _np.array([elem.item() if isinstance(elem, _np.ndarray) and elem.ndim == 0 else elem 
                                         for elem in v], dtype=object)
            except:
                pass  # If check fails, proceed normally
            return v
        
        # Handle multi-dim arrays (convert rows to objects)
        if len(v) == n:
            out = _np.empty(n, dtype=object)
            for i in range(n):
                row = v[i]
                # Unwrap 0-d nested arrays
                if isinstance(row, _np.ndarray) and row.ndim == 0:
                    out[i] = row.item()
                else:
                    out[i] = row
            return out
        
        # Length mismatch - pad/trim
        if len(v) == 1:
            return _np.repeat(v, n, axis=0)
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
            print(f"Failed to flush augmentation buffer: {e}")

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
            print(f"Failed to flush annotation buffer: {e}")

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
            print(f"Failed to flush pool snapshot buffer: {e}")

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
# Prioritized Sampling Enabled Query Pool
# ==============================

class PrioritizedQueryPool:
    """
    Prioritized sampling pool: sample problems proportional to 1-s_i where s_i is success rate.
    Success rate = normalized average reward over all rollouts of that problem.
    
    Maintains:
    - _records: All scored queries
    - _success_rates: Per-query average normalized reward (0=worst, 1=best)
    - _cold_queue: Unscored queries (sampled first to get initial rates)
    """
    
    def __init__(
        self,
        max_size: int = 30000,
        epsilon: float = 0.1,        # Smoothing to prevent zero weights
        temperature: float = 1.0,     # Sampling sharpness (higher=more uniform)
        rng: Optional[np.random.Generator] = None,
        trainer_ref: Optional["RayDAPOTrainer"] = None,
    ):
        self._lock = threading.RLock()
        self._max_size = max(1, int(max_size))
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)
        self.trainer_ref = trainer_ref
        
        # Core storage
        self._records: Dict[str, QueryRecord] = {}
        self._success_rates: Dict[str, float] = {}  # ∈ [0, 1]
        self._sample_counts: Dict[str, int] = {}
        self._cold_queue: deque[QueryRecord] = deque()
        self._cold_index: Dict[str, bool] = {}  # Track which record_ids are in cold queue
        
        self._rng = rng if rng is not None else np.random.default_rng()
        
        # Metrics
        self._total_added = 0
        self._total_sampled = 0
        self._total_evicted = 0
    
    # ==================== Admin ====================
    
    def set_max_size(self, max_size: int):
        with self._lock:
            old_size = self._max_size
            self._max_size = max(1, int(max_size))
            if self._max_size < old_size:
                self._evict_to_capacity_unlocked()
            logger.info(f"Pool max size changed from {old_size} to {self._max_size}")
    
    def size(self) -> int:
        with self._lock:
            return len(self._records) + len(self._cold_queue)
    
    def capacity_remaining(self) -> int:
        with self._lock:
            return max(0, self._max_size - self.size())
    
    def get_metrics(self) -> Dict[str, Any]:
        with self._lock:
            success_rates_list = list(self._success_rates.values())
            sample_counts_list = list(self._sample_counts.values())
            
            metrics = {
                "total_size": self.size(),
                "scored_size": len(self._records),
                "cold_queue_size": len(self._cold_queue),
                "capacity_remaining": self.capacity_remaining(),
                "total_added": self._total_added,
                "total_sampled": self._total_sampled,
                "total_evicted": self._total_evicted,
            }
            
            # Success rate statistics
            if success_rates_list:
                metrics["avg_success_rate"] = float(np.mean(success_rates_list))
                metrics["min_success_rate"] = float(np.min(success_rates_list))
                metrics["max_success_rate"] = float(np.max(success_rates_list))
                metrics["std_success_rate"] = float(np.std(success_rates_list))
            else:
                metrics["avg_success_rate"] = 0.0
                metrics["min_success_rate"] = 0.0
                metrics["max_success_rate"] = 1.0
                metrics["std_success_rate"] = 0.0
            
            # Sample count statistics (how many times each problem was scored)
            if sample_counts_list:
                metrics["avg_sample_count"] = float(np.mean(sample_counts_list))
                metrics["min_sample_count"] = int(np.min(sample_counts_list))
                metrics["max_sample_count"] = int(np.max(sample_counts_list))
            
            return metrics
    
    # ==================== Adding ====================
    
    def add_many(self, items: List[QueryRecord]):
        """Add multiple records. Unscored go to cold queue, scored go to main pool."""
        if not items:
            return
        
        print(f"\n{'='*70}")
        print(f"[PRIO-POOL-ADD] Adding {len(items)} items")
        print(f"{'='*70}")
        
        with self._lock:
            before_size = self.size()
            added_cold = 0
            added_scored = 0
            updated_existing = 0
            rejected = 0
            
            for it in items:
                # Validation
                if not self._validate_record(it):
                    rejected += 1
                    continue
                
                # Check capacity before adding NEW items
                if it.record_id not in self._records and self.size() >= self._max_size:
                    # Pool full, evict worst (highest success rate = easiest = least useful)
                    self._evict_one_unlocked()
                
                # Route based on scoring status
                if it.reward is None or not np.isfinite(it.reward):
                    # Unscored → cold queue
                    if it.record_id not in self._records and it.record_id not in self._cold_index:
                        self._cold_queue.append(it)
                        self._cold_index[it.record_id] = True  # Track in index
                        added_cold += 1
                        self._total_added += 1
                else:
                    # Scored → main pool (or update if exists)
                    if it.record_id in self._records:
                        # Update existing record
                        self._records[it.record_id] = it
                        # Keep existing success rate (will be updated during training)
                        updated_existing += 1
                    else:
                        # New scored record
                        self._records[it.record_id] = it
                        
                        # Don't overwrite success_rate if already loaded from checkpoint
                        if it.record_id not in self._success_rates:
                            norm_reward = self._normalize_reward(it.reward)
                            self._success_rates[it.record_id] = norm_reward
                        
                        if it.record_id not in self._sample_counts:
                            self._sample_counts[it.record_id] = 0
                        
                        added_scored += 1
                        self._total_added += 1
            
            after_size = self.size()
            
            print(f"  Results:")
            print(f"    Added to cold queue: {added_cold}")
            print(f"    Added to scored pool: {added_scored}")
            print(f"    Updated existing: {updated_existing}")
            print(f"    Rejected: {rejected}")
            print(f"    Pool size: {before_size} → {after_size} (Δ{after_size - before_size:+d})")
            print(f"{'='*70}\n")
    
    def _validate_record(self, rec: QueryRecord) -> bool:
        """Quick validation check."""
        try:
            data = rec.raw_prompt_data
            if not isinstance(data, np.ndarray):
                return False
            if data.ndim != 1:
                return False
            if data.dtype == object or not np.issubdtype(data.dtype, np.integer):
                return False
            if data.size == 0:
                return False
            return True
        except:
            return False
    
    def _evict_one_unlocked(self):
        """Evict the highest success rate item (easiest, least useful for learning)."""
        if not self._success_rates:
            # No scored items, evict from cold queue
            if self._cold_queue:
                evicted = self._cold_queue.popleft()
                self._cold_index.pop(evicted.record_id, None)
                self._cleanup_record(evicted)
                self._total_evicted += 1
                print(f"[EVICT] Removed cold item {evicted.record_id}")
            return
        
        # Find item with highest success rate (easiest)
        worst_id = max(self._success_rates.keys(), key=lambda k: self._success_rates[k])
        evicted_rate = self._success_rates[worst_id]
        
        # Remove from ALL data structures
        evicted_rec = self._records.pop(worst_id, None)
        self._success_rates.pop(worst_id, None)
        self._sample_counts.pop(worst_id, None)
        
        if evicted_rec:
            self._cleanup_record(evicted_rec)
            print(f"[EVICT] Removed scored item {worst_id} with success_rate={evicted_rate:.3f} (easiest)")
        
        self._total_evicted += 1
    
    def _evict_to_capacity_unlocked(self):
        """Evict items until within capacity."""
        while self.size() > self._max_size:
            self._evict_one_unlocked()
    
    def _cleanup_record(self, rec: QueryRecord):
        """Clean up record from trainer's index."""
        if rec is None:
            return
        try:
            if hasattr(self, 'trainer_ref') and self.trainer_ref is not None:
                trainer = self.trainer_ref
                if hasattr(trainer, '_record_index') and hasattr(trainer, '_inbox_lock'):
                    with trainer._inbox_lock:
                        trainer._record_index.pop(rec.record_id, None)
        except Exception:
            pass
    
    # ==================== Sampling ====================
    
    def sample_batch(self, k: int) -> List[QueryRecord]:
        """
        Sample k items:
        1. Cold queue first (need scoring)
        2. Then prioritized by (1 - success_rate)
        """
        if k <= 0:
            return []
        
        with self._lock:
            if self.size() == 0:
                return []
            
            actual_k = min(k, self.size())
            
            # Phase 1: Cold items (always prioritize)
            cold_take = min(actual_k, len(self._cold_queue))
            cold = [self._cold_queue.popleft() for _ in range(cold_take)]
            # Update cold index
            for rec in cold:
                self._cold_index.pop(rec.record_id, None)
            
            # Phase 2: Prioritized sampling
            need = actual_k - cold_take
            scored = []
            
            if need > 0 and self._records:
                # Compute sampling weights
                record_ids = list(self._records.keys())
                weights = []
                
                for rid in record_ids:
                    s_i = self._success_rates.get(rid, 0.5)  # Success rate ∈ [0, 1]
                    weight = 1.0 - s_i  # Higher weight for lower success (harder problems)
                    weights.append(weight)
                
                weights = np.array(weights, dtype=float)
                
                # Smoothing: add epsilon to prevent zero weights
                weights = weights + self.epsilon
                
                # Temperature adjustment (optional)
                if self.temperature != 1.0:
                    weights = np.power(weights, 1.0 / self.temperature)
                
                # Normalize to probabilities
                probs = weights / weights.sum()
                
                # Sample without replacement
                sampled_indices = self._rng.choice(
                    len(record_ids),
                    size=min(need, len(record_ids)),
                    replace=False,
                    p=probs
                )
                
                for idx in sampled_indices:
                    rid = record_ids[idx]
                    rec = self._records[rid]
                    # Quick validation before returning
                    if self._validate_record(rec):
                        scored.append(rec)
                    else:
                        print(f"[PRIO-POOL-SAMPLE] Skipping invalid record {rid} during sampling")

            chosen = cold + scored
            self._total_sampled += len(chosen)
            
            # Log sampling stats
            if chosen:
                cold_count = len(cold)
                scored_count = len(scored)
                
                if scored:
                    scored_rates = [self._success_rates.get(r.record_id, 0.5) for r in scored]
                    print(f"[PRIO-POOL-SAMPLE] Sampled {len(chosen)} items: {cold_count} cold, {scored_count} scored")
                    print(f"  Scored success rates: mean={np.mean(scored_rates):.3f}, "
                          f"min={np.min(scored_rates):.3f}, max={np.max(scored_rates):.3f}")
                else:
                    print(f"[PRIO-POOL-SAMPLE] Sampled {len(chosen)} items: {cold_count} cold, {scored_count} scored")
            
            # Return copies to prevent external modification
            return [self._copy_record_unlocked(r) for r in chosen]
    
    # ==================== Success Rate Updates ====================
    
    def update_success_rates(self, updates: List[Tuple[QueryRecord, float]]):
        """
        Update success rates based on new rollout results.
        
        Args:
            updates: List of (record, new_reward) tuples
        """
        with self._lock:
            promoted_count = 0
            updated_count = 0
            
            for record, new_reward in updates:
                record_id = record.record_id
                norm_reward = self._normalize_reward(new_reward)
                
                # Update the record itself
                record.reward = new_reward
                record.est_reward = new_reward
                
                if record_id in self._records:
                    # Existing scored record - update running average
                    old_count = self._sample_counts[record_id]
                    old_rate = self._success_rates[record_id]
                    
                    new_count = old_count + 1
                    new_rate = (old_rate * old_count + norm_reward) / new_count
                    new_rate = float(np.clip(new_rate, 0.0, 1.0))
                    
                    self._success_rates[record_id] = new_rate
                    self._sample_counts[record_id] = new_count
                    
                    # Update stored record
                    self._records[record_id] = record
                    updated_count += 1
                else:
                    # Was from cold queue - promote to scored pool
                    self._records[record_id] = record
                    self._success_rates[record_id] = norm_reward
                    self._sample_counts[record_id] = 1  # First scoring
                    promoted_count += 1
            
            if promoted_count > 0:
                print(f"[SUCCESS-RATE-UPDATE] Promoted {promoted_count} cold items to scored pool")
            if updated_count > 0:
                print(f"[SUCCESS-RATE-UPDATE] Updated {updated_count} existing scored items")
    
    @staticmethod
    def _normalize_reward(reward: float) -> float:
        """
        Normalize reward from [-1, 1] to [0, 1] for success rate.
        0 = worst, 1 = best
        """
        if not np.isfinite(reward):
            return 0.5  # Default to medium
        
        # Clip to expected range
        reward = float(np.clip(reward, -1.0, 1.0))
        
        # Linear mapping: [-1, 1] → [0, 1]
        normalized = (reward + 1.0) / 2.0
        
        return float(np.clip(normalized, 0.0, 1.0))
    
    # ==================== Utilities ====================
    
    @staticmethod
    def _copy_record_unlocked(it: QueryRecord) -> QueryRecord:
        return QueryRecord(
            raw_prompt_data=it.raw_prompt_data.copy() if isinstance(it.raw_prompt_data, np.ndarray) else it.raw_prompt_data,
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
            
            # Sample from cold queue
            for i in range(min(n // 2, len(self._cold_queue))):
                samples.append(self._cold_queue[i].to_dict())
            
            # Sample from scored pool
            if self._records:
                sample_ids = self._rng.choice(
                    list(self._records.keys()),
                    size=min(n // 2, len(self._records)),
                    replace=False
                )
                for rid in sample_ids:
                    rec_dict = self._records[rid].to_dict()
                    rec_dict['success_rate'] = self._success_rates.get(rid, 0.5)
                    rec_dict['sample_count'] = self._sample_counts.get(rid, 0)
                    samples.append(rec_dict)
            
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
      - OPTIONAL: immediate integration of teacher-annotated items into the pool
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
        augmentation_logger: Optional["AugmentationLogger"] = None,
        immediate_release: bool = True,  # <— set True to bypass any delay/buffer
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
        self.immediate_release = bool(immediate_release)

        # Metrics (local counters)
        self._processed_count = 0
        self._error_count = 0
        self._api_call_count = 0

        # OpenAI client (consider reading from env instead of hardcoding)
        api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
        self.client = OpenAI(api_key=api_key, timeout=api_timeout)

        # Thread pool for parallel API calls
        self.executor = ThreadPoolExecutor(max_workers=1)

    # ------------------------ small utilities ------------------------

    def _bump_aug_metric(self, key: str, delta: int) -> None:
        """Safely bump augmentation-related metrics on trainer."""
        try:
            with self.trainer_ref._metrics_lock:
                m = self.trainer_ref._augmentation_metrics
                m[key] = int(m.get(key, 0)) + int(delta)
                # keep success rate in sync when possible
                if key in ("total_teacher_integrated", "total_teacher_submitted"):
                    sub = int(m.get("total_teacher_submitted", 0))
                    integ = int(m.get("total_teacher_integrated", 0))
                    m["augmentation_success_rate"] = (float(integ) / float(sub)) if sub > 0 else 0.0
        except Exception:
            # metric updates must never crash the annotator
            pass

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        """Return True if `text` contains CJK or full-width punctuation."""
        if not text:
            return False
        return bool(re.search(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]', text))

    def enqueue_aug(self, aug_proto: "DataProto") -> bool:
        """Enqueue with backpressure. Returns False if queue is full."""
        try:
            self.queue.put(aug_proto, block=False)
            logger.debug(
                f"Enqueued augmentation proto (queue size={self.queue.qsize()})")
            return True
        except Full:
            print("Teacher annotation queue is full, applying backpressure")
            return False

    def shutdown(self):
        """Graceful shutdown with timeout."""
        with self._shutdown_lock:
            if not self._running:
                return
            self._running = False
            self.stop_event.set()
            # drain a bit
            timeout = 5.0
            start = time.time()
            while not self.queue.empty() and (time.time() - start) < timeout:
                time.sleep(0.1)
            self.executor.shutdown(wait=True)
            if not self.queue.empty():
                print(f"Shutting down with {self.queue.qsize()} items still in queue")

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
        """Return (content, finish_reason). Works across SDK variants."""
        try:
            if hasattr(response, "choices") and response.choices:
                ch = response.choices[0]
                fr = getattr(ch, "finish_reason", None)
                if fr is None:
                    fd = getattr(ch, "finish_details", None)
                    fr = getattr(fd, "type", None)
                    if fr is None and isinstance(fd, dict):
                        fr = fd.get("type")
                msg = getattr(ch, "message", None)
                if msg is None:
                    return "", fr
                parsed = getattr(msg, "parsed", None)
                if parsed:
                    try:
                        return json.dumps(parsed, ensure_ascii=False), fr
                    except Exception:
                        pass
                if hasattr(msg, "content"):
                    return (msg.content or "").strip(), fr
                if isinstance(msg, dict):
                    return (msg.get("content", "") or "").strip(), fr
        except Exception as e:
            print(f"Error extracting text/reason: {e}")
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
                        {"role": "system", "content": "You are a careful math data cleaner and solver."},
                        {"role": "user", "content": self._make_extract_and_solve_prompt(original, generation)},
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

    # ------------------------ main loop ------------------------

    def run(self):
        """Main annotation loop with proper synchronization."""
        import sys
        sys.stdout.reconfigure(line_buffering=True)
        
        logger.info(f"[TEACHER-DIAG-2] Teacher thread starting, poll_interval={self.poll_interval}")
        # print(f"[TEACHER-DIAG-2] Teacher thread START", flush=True)
        
        while self._running:
            try:
                logger.debug(f"[TEACHER-DIAG-2] Waiting for queue item...")
                aug_proto = self.queue.get(timeout=self.poll_interval)
                logger.info(f"[TEACHER-DIAG-2] Received item from queue, queue_size={self.queue.qsize()}")
                # print(f"[TEACHER-DIAG-2] Received item, queue={self.queue.qsize()}", flush=True)
            except Empty:
                if self.stop_event.is_set():
                    break
                continue

            if not self._running:
                self.queue.task_done()
                break

            try:
                # print(f"[TEACHER-DIAG-2] Processing item, queue={self.queue.qsize()}", flush=True)
                self._process_augmented_proto(aug_proto)
                self._processed_count += 1
                # print(f"[TEACHER-DIAG-2] Processed item, queue={self.queue.qsize()}", flush=True)
            except Exception as e:
                # print(f"[TEACHER-DIAG-2] Error processing item: {e}", flush=True)
                print(f"Error processing augmented proto: {e}")
                self._error_count += 1
            finally:
                self.queue.task_done()

    # ------------------------ immediate integration helpers ------------------------

    def _proto_to_query_records(self, dp: DataProto) -> List[QueryRecord]:
        """Convert DataProto to QueryRecords with comprehensive tracking."""
        print(f"\n[CONVERT-FLOW-1] PROTO_TO_QUERY_RECORDS")
        print(f"  Input batch size: {len(dp.batch.get('input_ids', []))}")
        
        recs: List[QueryRecord] = []

        if "input_ids" not in dp.batch:
            print("DataProto missing input_ids")
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
        teacher_responses = safe_get_array("teacher/raw_responses", None)
        teacher_diffs = safe_get_array("teacher/difficulty", 1.0)
        parent_ids = safe_get_array("parent_record_id", None)
        record_ids = safe_get_array("record_ids", None)
        policy_diffs = safe_get_array("difficulty_factors", 1.0)
        
        # Get readable text for display purposes (NOT for raw_prompt_data)
        augmented_text_readable = safe_get_array("cleaned_text_readable", None)
        if not augmented_text_readable or all(x is None for x in augmented_text_readable):
            augmented_text_readable = safe_get_array("augmented_text_readable", None)

        validation_stats = {
            "total": n,
            "unsolvable": 0,
            "extraction_failed": 0,
            "dtype_issues": 0,
            "validation_failed": 0,
            "success": 0,
        }

        for i in range(n):
            # Respect solvability flag
            if not bool(solv_list[i]):
                validation_stats["unsolvable"] += 1
                # Still log unsolvable items for analysis
                if self.augmentation_logger:
                    self.augmentation_logger.log_annotation({
                        "original_text": str(original_texts[i]) if original_texts[i] else None,
                        "augmented_text": str(augmented_text_readable[i]) if augmented_text_readable and augmented_text_readable[i] else None,
                        "solvable": False,
                        "reason": "marked_unsolvable_by_teacher",
                        "estimated_reward": float(est_list[i]) if est_list[i] is not None else None,
                    })
                continue

            # ===== CRITICAL FIX: ALWAYS derive from batch tensors =====
            try:
                ids_tensor = dp.batch["input_ids"][i]
                
                # Ensure 1-D
                if ids_tensor.dim() > 1:
                    print(f"[CONVERT-WARN] Item {i}: Squeezing {ids_tensor.dim()}-D tensor")
                    ids_tensor = ids_tensor.squeeze()
                
                if ids_tensor.dim() != 1:
                    print(f"[CONVERT-ERROR] Item {i}: Cannot squeeze to 1-D (shape: {ids_tensor.shape})")
                    validation_stats["extraction_failed"] += 1
                    continue
                
                # Convert to numpy
                token_ids_np = ids_tensor.detach().cpu().numpy()
                
                # Validate dtype BEFORE any potential corruption
                if token_ids_np.dtype == np.object_:
                    print(f"[BUG DETECTED] Item {i}: input_ids tensor produced object array!")
                    print(f"  Tensor dtype: {ids_tensor.dtype}, shape: {ids_tensor.shape}")
                    print(f"  Numpy dtype: {token_ids_np.dtype}, shape: {token_ids_np.shape}")
                    validation_stats["dtype_issues"] += 1
                    
                    # Try to recover by converting
                    try:
                        token_ids_np = token_ids_np.astype(np.int64)
                        print(f"  Recovery: Converted to {token_ids_np.dtype}")
                    except Exception as e:
                        print(f"  Recovery failed: {e}")
                        validation_stats["extraction_failed"] += 1
                        continue
                
                # Force proper integer dtype if needed
                if not np.issubdtype(token_ids_np.dtype, np.integer):
                    print(f"[CONVERT-WARN] Item {i}: Converting {token_ids_np.dtype} to int64")
                    token_ids_np = token_ids_np.astype(np.int64)
                
                # Final validation
                assert token_ids_np.ndim == 1, f"Expected 1-D, got {token_ids_np.ndim}-D"
                assert np.issubdtype(token_ids_np.dtype, np.integer), f"Expected integer dtype, got {token_ids_np.dtype}"
                assert token_ids_np.size > 0, "Empty array"
                
                # Debug first few items
                if i < 3:
                    print(f"  Item {i} token array: dtype={token_ids_np.dtype}, shape={token_ids_np.shape}, size={token_ids_np.size}")
                
            except Exception as e:
                print(f"[CONVERT-ERROR] Failed to extract token IDs for item {i}: {e}")
                validation_stats["extraction_failed"] += 1
                continue

            # Safe reward extraction
            # For prioritized sampling, don't estimate rewards for teacher items
            # Let them be scored naturally during rollouts (they'll go to cold queue)
            final_reward = None  # Will be added to cold queue

            # Get readable text for display
            augmented_text_str = None
            if augmented_text_readable and i < len(augmented_text_readable) and augmented_text_readable[i]:
                augmented_text_str = str(augmented_text_readable[i])
            else:
                # Fallback: decode from tokens
                try:
                    augmented_text_str = self.trainer_ref._decode_tokens_to_text(token_ids_np)
                except Exception as e:
                    logger.debug(f"Failed to decode augmented text: {e}")
                    augmented_text_str = None

            # Get position_ids safely
            position_ids = None
            if "position_ids" in dp.batch and i < len(dp.batch["position_ids"]):
                position_ids = dp.batch["position_ids"][i]

            record = QueryRecord(
                raw_prompt_data=token_ids_np,  # ✅ Now guaranteed to be int64 array
                input_ids=dp.batch["input_ids"][i],
                attention_mask=dp.batch["attention_mask"][i],
                position_ids=position_ids,
                gt=gt_list[i],
                reward=final_reward,
                est_reward=final_reward,
                original_text=str(original_texts[i]) if original_texts[i] else None,
                augmented_text=augmented_text_str,
                teacher_response=str(teacher_responses[i]) if teacher_responses[i] else None,
                record_id=(str(record_ids[i]) if record_ids[i] else str(uuid.uuid4())),
                meta={
                    "source": "math_dapo",
                    "origin": "augmented",
                    "solvable": bool(solv_list[i]),
                    "global_step": getattr(self, 'global_steps', 0),
                    "teacher_difficulty": float(teacher_diffs[i]) if teacher_diffs[i] is not None else 1.0,
                    "parent_id": str(parent_ids[i]) if parent_ids[i] else None,
                    "policy_difficulty": (float(policy_diffs[i]) if policy_diffs[i] is not None else 1.0),
                },
            )
            
            # Immediate post-creation validation
            try:
                data = record.raw_prompt_data
                if not isinstance(data, np.ndarray):
                    print(f"[POST-CREATE-ERROR] Item {i}: raw_prompt_data is {type(data)}, not ndarray!")
                    validation_stats["validation_failed"] += 1
                    continue
                elif data.dtype == np.object_:
                    print(f"[POST-CREATE-ERROR] Item {i}: raw_prompt_data has object dtype after creation!")
                    print(f"  This should never happen - bug in QueryRecord constructor?")
                    validation_stats["validation_failed"] += 1
                    continue
                elif not np.issubdtype(data.dtype, np.integer):
                    print(f"[POST-CREATE-ERROR] Item {i}: raw_prompt_data has dtype {data.dtype}, not integer!")
                    validation_stats["validation_failed"] += 1
                    continue
            except Exception as e:
                print(f"[POST-CREATE-ERROR] Item {i}: Validation exception: {e}")
                validation_stats["validation_failed"] += 1
                continue
            
            # Final validation pass
            if not self.trainer_ref._validate_and_fix_record(record):
                print(f"Skipping corrupted record that couldn't be fixed")
                validation_stats["validation_failed"] += 1
                continue
                
            validation_stats["success"] += 1
            recs.append(record)

        print(f"  Conversion stats:")
        print(f"    Total items: {validation_stats['total']}")
        print(f"    Unsolvable: {validation_stats['unsolvable']}")
        print(f"    Extraction failed: {validation_stats['extraction_failed']}")
        print(f"    Dtype issues: {validation_stats['dtype_issues']}")
        print(f"    Validation failed: {validation_stats['validation_failed']}")
        print(f"    Success: {validation_stats['success']}")
        print(f"[CONVERT-FLOW-1] END\n")
        
        return recs

    def _integrate_immediately(self, solvable_proto: "DataProto") -> bool:
        """Insert teacher items straight into the query pool, bypassing any delay."""
        if not hasattr(self.trainer_ref, "query_pool") or self.trainer_ref.query_pool is None:
            return False
        try:
            records = self._proto_to_query_records(solvable_proto)
            
            if not records:
                print("[Teacher] No valid records after conversion")
                return False
            
            # Register in lineage tracking
            if hasattr(self.trainer_ref, '_register_records'):
                self.trainer_ref._register_records(records)
            
            before = self.trainer_ref.query_pool.size()
            self.trainer_ref.query_pool.add_many(records)
            after = self.trainer_ref.query_pool.size()
            
            # ✅ FIX: Count actual additions, not attempted additions
            actual_added = after - before
            self._bump_aug_metric("total_teacher_integrated", actual_added)
            
            logger.info(f"[Teacher] Integrated {actual_added}/{len(records)} items immediately (pool {before} → {after})")
            
            if actual_added == 0:
                logger.warning(f"⚠️  Pool rejected all {len(records)} teacher records! Check validation.")
            elif actual_added < len(records):
                logger.warning(f"⚠️  Pool accepted {actual_added}/{len(records)} records ({len(records)-actual_added} rejected/evicted)")
            
            return actual_added > 0
        except Exception as e:
            print(f"Immediate integration failed; will fall back to submit_teacher_batch. {e}")
            import traceback
            traceback.print_exc()
            return False

    # ------------------------ core processing ------------------------

    def _process_augmented_proto(self, aug_proto: "DataProto"):
        """Process a single augmented proto with comprehensive logging."""
        
        # Filtering stats
        flow_stats = {
            "input_total": 0,
            "has_est_reward": 0,
            "filtered_chinese": 0,
            "scheduled_api_calls": 0,
            "api_failed": 0,
            "api_succeeded": 0,
            "no_clean_question": 0,
            "unsolvable_no_answer": 0,
            "marked_solvable": 0,
            "final_solvable_count": 0,
            "records_created": 0,
            "records_integrated": 0,
        }
        
        # Select only items that have an estimated reward
        est = aug_proto.non_tensor_batch.get("policy/est_reward", None)
        if est is None:
            # print(f"[TEACHER-FLOW-1] No est_reward field, SKIPPING")
            return
        
        flow_stats["input_total"] = len(list(est))
        # print(f"  Input items: {flow_stats['input_total']}")
        
        est = list(est)
        idxs = [i for i, v in enumerate(est) if self._valid_est(v)]
        flow_stats["has_est_reward"] = len(idxs)
        # print(f"  Items with valid est_reward: {flow_stats['has_est_reward']}")
        
        if not idxs:
            # print(f"[TEACHER-FLOW-1] No valid rewards, EXITING")
            return

        _sanitize_non_tensor_batch(aug_proto)
        sub_proto = aug_proto[idxs]

        # Get original texts (these should be text strings)
        original_texts = list(sub_proto.non_tensor_batch.get("original_text", []))
        
        # Get augmented texts - try multiple sources
        augmented_texts = None
        
        # Source 1: Check for readable text field (preferred)
        if "augmented_text_readable" in sub_proto.non_tensor_batch:
            augmented_texts = list(sub_proto.non_tensor_batch.get("augmented_text_readable", []))
            print(f"  Using 'augmented_text_readable' field ({len(augmented_texts)} items)")
        
        # Source 2: Check for cleaned_text_readable (from previous teacher pass)
        elif "cleaned_text_readable" in sub_proto.non_tensor_batch:
            augmented_texts = list(sub_proto.non_tensor_batch.get("cleaned_text_readable", []))
            print(f"  Using 'cleaned_text_readable' field ({len(augmented_texts)} items)")
        
        # Source 3: Decode from raw_prompt_data (token IDs)
        else:
            print(f"  No text field found, decoding from raw_prompt_data...")
            raw_data = list(sub_proto.non_tensor_batch.get("raw_prompt_data", []))
            augmented_texts = []
            decode_failures = 0
            
            for i, data in enumerate(raw_data):
                try:
                    # Check if it's already text
                    if isinstance(data, str):
                        augmented_texts.append(data)
                    # Check if it's token IDs
                    elif isinstance(data, np.ndarray):
                        # Decode tokens to text
                        text = self.trainer_ref._decode_tokens_to_text(data)
                        augmented_texts.append(text)
                    else:
                        # Unknown format
                        augmented_texts.append(str(data))
                        decode_failures += 1
                except Exception as e:
                    print(f"    WARNING: Failed to decode item {i}: {e}")
                    augmented_texts.append("")
                    decode_failures += 1
            
            print(f"  Decoded {len(augmented_texts)} items ({decode_failures} failures)")
        
        # Ensure lists are properly sized
        original_texts = original_texts[:len(idxs)] + [None] * max(0, len(idxs) - len(original_texts))
        augmented_texts = augmented_texts[:len(idxs)] + [None] * max(0, len(idxs) - len(augmented_texts))
        
        # Validate we have actual text
        sample_size = min(3, len(augmented_texts))
        text_issues = 0
        
        for i in range(sample_size):
            aug_text = str(augmented_texts[i]) if augmented_texts[i] else ""
            orig_text = str(original_texts[i]) if i < len(original_texts) and original_texts[i] else ""
            
        if text_issues > 0:
            print(f"  ⚠️  WARNING: {text_issues}/{sample_size} samples look like token IDs!")
            print(f"     Teacher will likely fail to extract clean questions.")
        
        raw_prompts = augmented_texts

        N = len(raw_prompts)
        teacher_answers = [None] * N
        teacher_solvable = [False] * N
        teacher_raw_responses = [""] * N
        clean_questions = ["" for _ in range(N)]
        teacher_difficulty = [1.0 for _ in range(N)]
        futures = []

        for i, q in enumerate(raw_prompts):
            if not self._running:
                break
            q_text = str(q) if q else ""
            
            # Chinese filter
            if self._contains_chinese(q_text):
                teacher_solvable[i] = False
                teacher_answers[i] = None
                teacher_raw_responses[i] = "[filtered:contains_chinese]"
                flow_stats["filtered_chinese"] += 1
                
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
            fut = self.executor.submit(self._call_extract_solve_with_retry, orig, q_text)
            futures.append((i, fut, q_text))

        flow_stats["scheduled_api_calls"] = len(futures)
        # print(f"  Filtered Chinese: {flow_stats['filtered_chinese']}")
        # print(f"  Scheduled API calls: {flow_stats['scheduled_api_calls']}")
        
        # Update metric
        self._bump_aug_metric("total_teacher_submitted", len(futures))

        # print(f"\n[TEACHER-FLOW-3] API CALL STAGE")
        # Collect results
        for idx, (i, future, question_text) in enumerate(futures):
            try:
                r = future.result(timeout=self.api_timeout * 2)
                
                if not r.get("ok", False):
                    flow_stats["api_failed"] += 1
                    teacher_solvable[i] = False
                    teacher_answers[i] = None
                    teacher_raw_responses[i] = r.get("raw", "")
                    
                    # Log sample failures (first 3)
                    if flow_stats["api_failed"] <= 3:
                        print(f"    [API FAIL {flow_stats['api_failed']}] Item {i}: {r.get('error', 'unknown')}")
                    continue
                
                flow_stats["api_succeeded"] += 1

                ans = r.get("answer")
                is_solvable = bool(r.get("solvable", False))
                has_ans = isinstance(ans, str) and ans.strip() != ""

                clean_questions[i] = r["clean"]
                teacher_difficulty[i] = r["difficulty"]
                teacher_raw_responses[i] = r["raw"]

                # No clean question extracted
                if not clean_questions[i]:
                    flow_stats["no_clean_question"] += 1
                    teacher_solvable[i] = False
                    teacher_answers[i] = None

                    continue
                
                # Marked solvable but no answer
                if is_solvable and not has_ans:
                    flow_stats["unsolvable_no_answer"] += 1
                    teacher_solvable[i] = False
                    teacher_answers[i] = None
                    teacher_raw_responses[i] = r.get("raw", "")
                    
                # SUCCESS
                teacher_solvable[i] = is_solvable
                teacher_answers[i] = ans if is_solvable else None
                
                if is_solvable:
                    flow_stats["marked_solvable"] += 1

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
                flow_stats["api_failed"] += 1
                teacher_solvable[i] = False
                teacher_answers[i] = None
                teacher_raw_responses[i] = "[timeout]"
            except Exception as e:
                flow_stats["api_failed"] += 1
                teacher_solvable[i] = False
                teacher_answers[i] = None
                teacher_raw_responses[i] = f"[error]{e}"

        # print(f"  API succeeded: {flow_stats['api_succeeded']}")
        # print(f"  API failed: {flow_stats['api_failed']}")
        # print(f"  No clean question: {flow_stats['no_clean_question']}")
        # print(f"  Solvable but no answer: {flow_stats['unsolvable_no_answer']}")
        # print(f"  Marked solvable: {flow_stats['marked_solvable']}")

        # Attach teacher results
        sub_proto.non_tensor_batch["teacher/gt"] = teacher_answers
        sub_proto.non_tensor_batch["teacher/solvable"] = np.array(teacher_solvable, dtype=bool)
        sub_proto.non_tensor_batch["teacher/raw_responses"] = teacher_raw_responses
        sub_proto.non_tensor_batch["raw_prompt_data"] = np.asarray(clean_questions, dtype=object)
        sub_proto.non_tensor_batch["teacher/difficulty"] = np.asarray(teacher_difficulty, dtype=float)

        # ============ WRAP IN INSTRUCTION TEMPLATE ============
        # print(f"\n[TEACHER-FLOW-3.5] WRAPPING IN INSTRUCTION TEMPLATE")
        
        INSTRUCTION_PREFIX = (
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            "Solve the following math problem step by step. "
            "The last line of your response should be of the form Answer: \$Answer (without quotes) "
            "where \$Answer is the answer to the problem.\n\n"
        )
        
        INSTRUCTION_SUFFIX = (
            "\n\nRemember to put your answer on its own line after \"Answer:\".<|im_end|>\n<|im_start|>assistant\n"
        )
        
        wrapped_questions = []
        for q in clean_questions:
            if q and q.strip():  # Only wrap non-empty questions
                wrapped = INSTRUCTION_PREFIX + q.strip() + INSTRUCTION_SUFFIX
                wrapped_questions.append(wrapped)
            else:
                wrapped_questions.append(q)  # Keep empty as-is
        
        clean_questions = wrapped_questions
        # print(f"  Wrapped {len(clean_questions)} questions")
        # ============ END WRAPPING ============

        # Retokenization
        # print(f"\n[TEACHER-FLOW-4] RETOKENIZATION")
        try:
            # enc = self.trainer_ref._tokenize_texts([[{"role": "user", "content": clean_q}] for clean_q in clean_questions])
            enc = self.trainer_ref._tokenize_texts(clean_questions)
            sub_proto.batch["input_ids"] = enc["input_ids"]
            sub_proto.batch["attention_mask"] = enc["attention_mask"]
            sub_proto.batch["position_ids"] = enc["position_ids"]
            print(f"  Retokenized {len(clean_questions)} items successfully")
        except Exception as e:
            print(f"  CRITICAL: Retokenization failed: {e}")
            print(f"[TEACHER-FLOW] ABORTING BATCH")
            print(f"{'='*70}\n")
            return

        # ===== FIX: Create proper 1-D integer arrays, NOT object arrays =====
        # print(f"\n[TEACHER-FLOW-4b] CONVERTING TO PROPER TOKEN ARRAYS")
        token_ids_list = []
        conversion_failures = 0
        
        for i in range(len(clean_questions)):
            try:
                # Extract tensor and convert to proper numpy array
                ids_tensor = enc["input_ids"][i]  # Shape: [L]
                ids_np = ids_tensor.detach().cpu().numpy()  # numpy array
                
                # Ensure it's 1-D integer array (not object array)
                if ids_np.ndim != 1:
                    print(f"  WARNING: Item {i} has ndim={ids_np.ndim}, squeezing")
                    ids_np = ids_np.squeeze()
                
                if not np.issubdtype(ids_np.dtype, np.integer):
                    print(f"  WARNING: Item {i} has dtype={ids_np.dtype}, converting to int64")
                    ids_np = ids_np.astype(np.int64)
                
                token_ids_list.append(ids_np)
                
            except Exception as e:
                print(f"  ERROR: Failed to convert item {i}: {e}")
                conversion_failures += 1
                # Create dummy array as fallback
                token_ids_list.append(np.array([0], dtype=np.int64))
        
        # print(f"  Converted {len(token_ids_list)} items ({conversion_failures} failures)")
        
        # Store as object array (unavoidable for variable-length sequences)
        # BUT each element is now a proper 1-D int64 array
        sub_proto.non_tensor_batch["raw_prompt_data"] = np.array(token_ids_list, dtype=object)
        
        # Also store human-readable text separately
        sub_proto.non_tensor_batch["cleaned_text_readable"] = np.asarray(clean_questions, dtype=object)

        _sanitize_non_tensor_batch(sub_proto)

        # Filter to solvable only
        keep_idxs = np.asarray([i for i, ok in enumerate(teacher_solvable) if ok], dtype=np.int64)
        flow_stats["final_solvable_count"] = len(keep_idxs)
        
        # print(f"\n[TEACHER-FLOW-5] FINAL FILTERING")
        # print(f"  Final solvable count: {flow_stats['final_solvable_count']}")
        
        if keep_idxs.size == 0:
            print(f"  No solvable items, EXITING")
            print(f"[TEACHER-FLOW] BATCH COMPLETE - ZERO OUTPUT")
            print(f"{'='*70}\n")
            return

        solvable_proto = sub_proto[keep_idxs]

        # Integration
        # print(f"\n[TEACHER-FLOW-6] INTEGRATION STAGE")
        # print(f"  immediate_release: {self.immediate_release}")
        
        if self.immediate_release:
            # print(f"  Attempting immediate integration...")
            pool_size_before = self.trainer_ref.query_pool.size()
            
            integrated = self._integrate_immediately(solvable_proto)
            
            pool_size_after = self.trainer_ref.query_pool.size()
            actual_added = pool_size_after - pool_size_before
            flow_stats["records_integrated"] = actual_added
            
            # print(f"  Integration result: {integrated}")
            # print(f"  Pool size: {pool_size_before} → {pool_size_after} (added {actual_added})")
            
            if integrated:
                # print(f"\n[TEACHER-FLOW] BATCH COMPLETE - SUCCESS")
                # print(f"  Summary:")
                # for k, v in flow_stats.items():
                #     print(f"    {k}: {v}")
                # print(f"{'='*70}\n")
                return

        # Fallback
        print(f"  Falling back to submit_teacher_batch...")
        if hasattr(self.trainer_ref, "submit_teacher_batch"):
            self.trainer_ref.submit_teacher_batch(solvable_proto)
            print(f"  Submitted to inbox")
        else:
            self._integrate_immediately(solvable_proto)
            print(f"  Last-resort immediate integration")
        
        print(f"\n[TEACHER-FLOW] BATCH COMPLETE")
        print(f"  Summary:")
        for k, v in flow_stats.items():
            print(f"    {k}: {v}")
        print(f"{'='*70}\n")

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
                print(
                    f"Failed to initialize tokenizer: {e}. Using default.")
                self.tokenizer = None

        # Initialize metrics tracking
        self._augmentation_metrics = {
            "total_augmented": 0,
            "total_teacher_submitted": 0,
            "total_teacher_integrated": 0,
            "augmentation_success_rate": 0.0,
            "avg_difficulty_factor": 0.0,
        }
        
        self._metrics_lock = threading.Lock()

        # Initialize augmentation logger for offline analysis
        log_dir = getattr(config.trainer, 'default_local_dir', './logs')
        experiment_name = getattr(
            config.trainer, 'experiment_name', 'dapo_experiment')
        self.augmentation_logger = AugmentationLogger(log_dir, experiment_name)

        # Track augmentation history for analysis
        self.augmentation_history = []
        self.max_history_size = 10000

    def _save_training_state(self, checkpoint_dir: str):
        """
        Save all dynamic training state for crash recovery.
        Handles TensorDict, numpy object arrays, and DataProto objects safely.
        """
        import gzip
        import time
        
        state_path = os.path.join(checkpoint_dir, "training_state.pt.gz")
        
        logger.info(f"Saving training state to {state_path}...")
        save_start = time.time()
        
        try:
            # Serialize all records (CPU + validation)
            def _serialize_records_safe(records, label):
                serialized = []
                failed = 0
                for i, rec in enumerate(records):
                    try:
                        # Validate before serializing
                        if not self._is_record_valid_without_fix(rec):
                            logger.warning(f"  Skipping invalid {label} record {i}/{len(records)}")
                            failed += 1
                            continue
                        serialized.append(_serialize_query_record(rec))
                    except Exception as e:
                        logger.warning(f"  Failed to serialize {label} record {i}: {e}")
                        failed += 1
                if failed > 0:
                    logger.warning(f"  Failed to serialize {failed}/{len(records)} {label} records")
                return serialized
            

            with self.query_pool._lock:
                scored_records = list(self.query_pool._records.values())
                cold_records = list(self.query_pool._cold_queue)
                success_rates = dict(self.query_pool._success_rates)
                sample_counts = dict(self.query_pool._sample_counts)
                
                pool_metrics = {
                    'total_added': self.query_pool._total_added,
                    'total_sampled': self.query_pool._total_sampled,
                    'total_evicted': self.query_pool._total_evicted,
                }

            print(f"  Extracted {len(scored_records)} scored, {len(cold_records)} cold records")

            # Serialize
            scored_serialized = [_serialize_query_record(r) for r in scored_records]
            cold_serialized = [_serialize_query_record(r) for r in cold_records]
            
            # Extract inbox (thread-safe)
            with self._inbox_lock:
                inbox_protos = list(self._annotated_inbox)

            print(f"  Extracted {len(inbox_protos)} inbox protos")
            
            # 4. Serialize trained archive and record index
            logger.info(f"  Serializing trained archive ({len(self._trained_archive)} items)...")
            trained_archive_serialized = {}
            for k, v in self._trained_archive.items():
                try:
                    if self._is_record_valid_without_fix(v):
                        trained_archive_serialized[k] = _serialize_query_record(v)
                except Exception as e:
                    logger.debug(f"Failed to serialize archive record {k}: {e}")
            
            logger.info(f"  Serializing record index ({len(self._record_index)} items)...")
            record_index_serialized = {}
            for k, v in self._record_index.items():
                try:
                    if self._is_record_valid_without_fix(v):
                        record_index_serialized[k] = _serialize_query_record(v)
                except Exception as e:
                    logger.debug(f"Failed to serialize index record {k}: {e}")
            
            # 5. Serialize inbox DataProtos
            logger.info("  Serializing inbox DataProtos...")
            inbox_serialized = []
            for i, dp in enumerate(inbox_protos):
                try:
                    inbox_serialized.append(_serialize_dataproto(dp))
                except Exception as e:
                    logger.warning(f"  Failed to serialize inbox proto {i}: {e}")
            
            # 6. Serialize seed records template
            logger.info(f"  Serializing seed template ({len(self._seed_records_template)} items)...")
            seed_template_serialized = _serialize_records_safe(self._seed_records_template, "seed")
            
            # 7. Build complete state dict
            state = {
                'version': 2,  # Increment version for new serialization format
                'save_timestamp': time.time(),
                'global_steps': self.global_steps,
                'gen_steps': getattr(self, 'gen_steps', self.global_steps),
                'current_epoch': self.current_epoch,
                
                'query_pool': {
                    'scored_records': scored_serialized,
                    'cold_records': cold_serialized,
                    'success_rates': success_rates,
                    'sample_counts': sample_counts,
                    'cold_index': dict(self.query_pool._cold_index),
                    'max_size': self.query_pool._max_size,
                    'epsilon': self.query_pool.epsilon,
                    'temperature': self.query_pool.temperature,
                    'metrics': pool_metrics,
                },
                
                'trained_archive': trained_archive_serialized,
                'record_index': record_index_serialized,
                
                'lineage': {
                    'parent_to_children': {k: list(v) for k, v in self._parent_to_children.items()},
                    'child_to_parent': dict(self._child_to_parent),
                },
                
                'seed_records_template': seed_template_serialized,
                'reseed_round': self._reseed_round,
                
                'inbox_protos': inbox_serialized,
                
                'augmentation_metrics': dict(self._augmentation_metrics),
            }
            
            # 8. Save with compression
            logger.info("  Writing to disk with gzip compression...")
            with gzip.open(state_path, 'wb', compresslevel=6) as f:
                torch.save(state, f)
            
            # 9. Report statistics
            save_time = time.time() - save_start
            file_size_mb = os.path.getsize(state_path) / (1024 * 1024)
            
            logger.info(f"✓ Training state saved in {save_time:.1f}s ({file_size_mb:.1f} MB)")
            logger.info(f"  Pool: {len(scored_serialized)} scored, {len(cold_serialized)} cold")
            logger.info(f"  Archive: {len(trained_archive_serialized)} items")
            logger.info(f"  Index: {len(record_index_serialized)} items")
            logger.info(f"  Inbox: {len(inbox_serialized)} protos")
            logger.info(f"  Seed template: {len(seed_template_serialized)} items")
            
        except Exception as e:
            logger.error(f"Failed to save training state: {e}")
            # Don't crash training, but warn loudly
            print(f"\n{'='*70}")
            print(f"WARNING: Training state save failed! Recovery will not be possible.")
            print(f"Error: {e}")
            print(f"{'='*70}\n")

    def _load_training_state(self, checkpoint_dir: str) -> bool:
        """
        Load training state for crash recovery.
        Returns True if successful, False if no state found.
        """
        import gzip
        import time
        
        state_path = os.path.join(checkpoint_dir, "training_state.pt.gz")
        
        if not os.path.exists(state_path):
            logger.info(f"No training state found at {state_path}, starting fresh")
            return False
        
        logger.info(f"Loading training state from {state_path}...")
        load_start = time.time()
        
        try:
            # 1. Load state file
            with gzip.open(state_path, 'rb') as f:
                state = torch.load(f, map_location='cpu', weights_only=False)

            version = state.get('version', 1)
            logger.info(f"  State version: {version}")
            
            if version > 2:
                raise ValueError(f"Unsupported state version {version} (current version: 2)")
            
            # 2. Restore global counters
            self.global_steps = state['global_steps']
            self.gen_steps = state.get('gen_steps', self.global_steps)
            self.current_epoch = state['current_epoch']
            self._reseed_round = state['reseed_round']
            
            logger.info(f"  Restored step counters: global={self.global_steps}, epoch={self.current_epoch}")
            
            # 3. Deserialize all records
            def _deserialize_records_safe(serialized_list, label):
                records = []
                failed = 0
                for i, ser in enumerate(serialized_list):
                    try:
                        rec = _deserialize_query_record(ser)
                        # Validate after deserializing
                        if not self._validate_and_fix_record(rec):
                            logger.warning(f"  Skipping corrupted {label} record {i}")
                            failed += 1
                            continue
                        records.append(rec)
                    except Exception as e:
                        logger.warning(f"  Failed to deserialize {label} record {i}: {e}")
                        failed += 1
                if failed > 0:
                    logger.warning(f"  Failed to load {failed}/{len(serialized_list)} {label} records")
                return records
            
            pool_state = state['query_pool']

            logger.info("  Deserializing pool records...")
            scored_records = _deserialize_records_safe(pool_state['scored_records'], "scored")
            cold_records = _deserialize_records_safe(pool_state['cold_records'], "cold")

            # Recreate query pool
            logger.info("  Recreating query pool...")
            self.query_pool = PrioritizedQueryPool(
                max_size=pool_state['max_size'],
                epsilon=pool_state.get('epsilon', 0.1),
                temperature=pool_state.get('temperature', 1.0),
                rng=self._rng,
                trainer_ref=self,
            )

            # Restore cold index (or rebuild from cold_records)
            saved_cold_index = pool_state.get('cold_index', {})
            if saved_cold_index:
                self.query_pool._cold_index = saved_cold_index
            else:
                # Rebuild from cold_records
                logger.info("  Rebuilding cold index from cold records...")
                for rec in cold_records:
                    self.query_pool._cold_index[rec.record_id] = True

            # Restore success rates and sample counts
            logger.info("  Restoring success rates...")
            self.query_pool._success_rates = pool_state.get('success_rates', {})
            self.query_pool._sample_counts = pool_state.get('sample_counts', {})

            # Re-add all records
            all_records = scored_records + cold_records
            if all_records:
                logger.info(f"  Adding {len(all_records)} records to pool...")
                self.query_pool.add_many(all_records)
            
            # Restore pool metrics
            self.query_pool._total_added = pool_state['metrics']['total_added']
            self.query_pool._total_sampled = pool_state['metrics']['total_sampled']
            self.query_pool._total_evicted = pool_state['metrics']['total_evicted']
            
            # 5. Restore archives and indexes
            logger.info("  Deserializing trained archive...")
            self._trained_archive = {}
            for k, ser in state['trained_archive'].items():
                try:
                    rec = _deserialize_query_record(ser)
                    if self._validate_and_fix_record(rec):
                        self._trained_archive[k] = rec
                except Exception as e:
                    logger.debug(f"Failed to load archive record {k}: {e}")
            
            logger.info("  Deserializing record index...")
            self._record_index = {}
            for k, ser in state['record_index'].items():
                try:
                    rec = _deserialize_query_record(ser)
                    if self._validate_and_fix_record(rec):
                        self._record_index[k] = rec
                except Exception as e:
                    logger.debug(f"Failed to load index record {k}: {e}")
            
            # 6. Restore lineage
            logger.info("  Restoring lineage graphs...")
            lineage = state['lineage']
            self._parent_to_children = defaultdict(set, {
                k: set(v) for k, v in lineage['parent_to_children'].items()
            })
            self._child_to_parent = lineage['child_to_parent']
            
            # 7. Restore seed template
            logger.info("  Deserializing seed template...")
            self._seed_records_template = _deserialize_records_safe(
                state['seed_records_template'], "seed_template"
            )
            
            # 8. Restore inbox
            logger.info("  Deserializing inbox DataProtos...")
            with self._inbox_lock:
                self._annotated_inbox.clear()
                for ser in state['inbox_protos']:
                    try:
                        dp = _deserialize_dataproto(ser)
                        if dp is not None:
                            self._annotated_inbox.append(dp)
                    except Exception as e:
                        logger.warning(f"Failed to load inbox proto: {e}")
            
            # 9. Restore metrics
            self._augmentation_metrics = state['augmentation_metrics']
            
            # 10. Report statistics
            load_time = time.time() - load_start
            logger.info(f"✓ Training state loaded in {load_time:.1f}s")
            logger.info(f"  Resuming from step {self.global_steps}, epoch {self.current_epoch}")
            logger.info(f"  Pool size: {self.query_pool.size()} items")
            logger.info(f"    └─ scored: {len(scored_records)}, cold: {len(cold_records)}")
            logger.info(f"  Trained archive: {len(self._trained_archive)} items")
            logger.info(f"  Record index: {len(self._record_index)} items")
            logger.info(f"  Lineage: {len(self._parent_to_children)} parents, {len(self._child_to_parent)} children")
            logger.info(f"  Seed template: {len(self._seed_records_template)} items")
            logger.info(f"  Inbox: {len(self._annotated_inbox)} protos pending")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load training state: {e}")
            print(f"\n{'='*70}")
            print(f"WARNING: Failed to load training state from {state_path}")
            print(f"Error: {e}")
            print(f"Will start fresh (you may lose augmented data)")
            print(f"{'='*70}\n")
            return False

    def _dump_origin_rewards(self, batch: DataProto, path: str, step: int) -> None:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            ds = list(batch.non_tensor_batch.get("data_source", []))
            rids = list(batch.non_tensor_batch.get("record_ids", []))
            origins = list(batch.non_tensor_batch.get("origin", []))
            seq_rewards = batch.batch["token_level_rewards"].sum(
                dim=-1).detach().cpu().tolist()
            driver_rewards = list(batch.non_tensor_batch.get(
                "driver_reward", [None]*len(seq_rewards)))

            with open(path, "a", encoding="utf-8") as f:
                for i in range(min(len(seq_rewards), len(ds), len(rids))):
                    origin = None
                    if i < len(origins) and origins[i] is not None:
                        origin = str(origins[i])
                    else:
                        origin = "augmented" if str(
                            ds[i]) == "math_dapo" else (str(ds[i]) or "train")
                    rec = {
                        "step": step,
                        "record_id": str(rids[i]),
                        "origin": origin,
                        "seq_final_reward": float(seq_rewards[i]),
                        "driver_reward": (float(driver_rewards[i]) if driver_rewards and driver_rewards[i] is not None else None),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Failed to dump origin/rewards: {e}")

    def _dump_debug_batch(self, batch: DataProto, step: int):
        """
        Dump the current batch prompts to JSONL for debugging generation failures.
        Appends to a continuous log file for historical tracking.
        
        Args:
            batch: DataProto containing the batch to dump
            step: Current training step number
        """
        try:
            output_dir = os.path.join(self.config.trainer.default_local_dir)
            os.makedirs(output_dir, exist_ok=True)
            
            # Append to a continuous log file (changed from overwrite mode)
            output_file = os.path.join(output_dir, "all_gen_batches.jsonl")
            
            records = []
            input_ids = batch.batch.get("input_ids")
            attention_mask = batch.batch.get("attention_mask")
            
            if input_ids is None:
                logger.warning("Cannot dump debug batch: no input_ids found")
                return
            
            for i in range(input_ids.size(0)):
                try:
                    # Get the tokens for this item
                    ids = input_ids[i].detach().cpu().tolist()
                    mask = attention_mask[i].detach().cpu().tolist() if attention_mask is not None else None
                    
                    # Decode to text
                    if self.tokenizer:
                        prompt_text = self.tokenizer.decode(ids, skip_special_tokens=False)
                    else:
                        prompt_text = str(ids)
                    
                    # Get actual length (non-padding)
                    actual_len = sum(mask) if mask else len(ids)
                    
                    record = {
                        "index": i,
                        "step": step,
                        "prompt": prompt_text,
                        "token_length": actual_len,
                        "max_length": len(ids),
                    }
                    
                    # Add optional metadata if available
                    if hasattr(batch, 'non_tensor_batch'):
                        if "record_ids" in batch.non_tensor_batch:
                            record_ids = batch.non_tensor_batch.get("record_ids", [])
                            if i < len(record_ids):
                                record["record_id"] = str(record_ids[i])
                        
                        if "origin" in batch.non_tensor_batch:
                            origins = batch.non_tensor_batch.get("origin", [])
                            if i < len(origins):
                                record["origin"] = str(origins[i])
                        
                        if "driver_reward" in batch.non_tensor_batch:
                            rewards = batch.non_tensor_batch.get("driver_reward", [])
                            if i < len(rewards):
                                try:
                                    record["reward"] = float(rewards[i])
                                except:
                                    pass
                    
                    records.append(record)
                except Exception as e:
                    logger.debug(f"Failed to process item {i} in debug dump: {e}")
                    continue
            
            # Write to file in APPEND mode (changed from 'w' to 'a')
            with open(output_file, 'a', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            print(f"[DEBUG-DUMP] Appended {len(records)} prompts to {output_file} (step {step})")
            
        except Exception as e:
            # Never crash training due to debug logging
            print(f"Failed to dump debug batch: {e}")

    def _save_rollout_records(self, batch: DataProto, step: int):
        """
        Save query-rollout pairs to JSONL, sorted by rollout length.
        
        Each record contains:
        - query: the input prompt text
        - rollout: the generated response text
        - answer: ground truth answer (if available)
        - reward: computed reward for this rollout
        - rollout_length: number of tokens in the rollout
        - query_length: number of tokens in the query
        
        Records are sorted by rollout length (shortest to longest).
        
        Args:
            batch: DataProto containing queries, rollouts, and rewards
            step: Current training step number
        """
        try:
            output_dir = os.path.join(self.config.trainer.default_local_dir, "rollouts")
            os.makedirs(output_dir, exist_ok=True)
            
            # Filename with step number for chronological ordering
            output_file = os.path.join(output_dir, f"step_{step:06d}_rollouts.jsonl")
            
            # === Extract sequences ===
            sequences = batch.batch.get("sequences", batch.batch.get("input_ids"))
            if sequences is None:
                logger.warning("Cannot save rollouts: no sequences found in batch")
                return
            
            attention_mask = batch.batch.get("attention_mask")
            response_mask = batch.batch.get("response_mask")
            
            # === Get original prompt IDs to determine prompt length for each item ===
            raw_prompt_ids = batch.non_tensor_batch.get("raw_prompt_ids")
            if raw_prompt_ids is None:
                # Fallback: try to use raw_prompt_data
                raw_prompt_data = batch.non_tensor_batch.get("raw_prompt_data")
                if raw_prompt_data is not None:
                    raw_prompt_ids = raw_prompt_data
            
            if raw_prompt_ids is None:
                print("Cannot save rollouts: no raw_prompt_ids or raw_prompt_data found")
                return
            
            # === Extract rewards ===
            if "token_level_rewards" in batch.batch:
                rewards = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
            elif "token_level_scores" in batch.batch:
                rewards = batch.batch["token_level_scores"].sum(dim=-1).detach().cpu().numpy()
            else:
                rewards = np.zeros(sequences.size(0))
            
            # === Extract ground truth answers ===
            reward_model_data = batch.non_tensor_batch.get("reward_model", [])
            ground_truths = []
            for i in range(sequences.size(0)):
                if i < len(reward_model_data) and isinstance(reward_model_data[i], dict):
                    gt = reward_model_data[i].get("ground_truth", None)
                    ground_truths.append(str(gt) if gt is not None else None)
                else:
                    # Fallback: try other common keys
                    gt_list = batch.non_tensor_batch.get("gt", [])
                    if i < len(gt_list):
                        ground_truths.append(str(gt_list[i]) if gt_list[i] is not None else None)
                    else:
                        ground_truths.append(None)
            
            # === Extract optional metadata ===
            record_ids = batch.non_tensor_batch.get("record_ids", [])
            origins = batch.non_tensor_batch.get("origin", [])
            uids = batch.non_tensor_batch.get("uid", [])
            
            # === Process each item in batch ===
            records = []
            for i in range(sequences.size(0)):
                try:
                    # Get full sequence
                    seq = sequences[i].detach().cpu()
                    attn = attention_mask[i].detach().cpu() if attention_mask is not None else None
                    
                    # Filter out padding using attention mask
                    if attn is not None:
                        valid_mask = (attn > 0)
                        seq = seq[valid_mask]
                    
                    if seq.numel() == 0:
                        continue  # Skip empty sequences
                    
                    # === Determine prompt length from raw_prompt_ids ===
                    prompt_ids = raw_prompt_ids[i] if i < len(raw_prompt_ids) else None
                    
                    if prompt_ids is None:
                        print(f"Warning: No prompt IDs for item {i}, skipping")
                        continue
                    
                    # Handle different formats of prompt_ids
                    if isinstance(prompt_ids, np.ndarray):
                        prompt_length = len(prompt_ids)
                    elif isinstance(prompt_ids, (list, tuple)):
                        prompt_length = len(prompt_ids)
                    elif isinstance(prompt_ids, torch.Tensor):
                        prompt_length = prompt_ids.numel()
                    else:
                        print(f"Warning: Unknown prompt_ids format for item {i}: {type(prompt_ids)}")
                        continue
                    
                    # Ensure prompt_length doesn't exceed sequence length
                    prompt_length = min(prompt_length, len(seq))
                    
                    # Separate query and rollout based on prompt length
                    query_ids = seq[:prompt_length].tolist()
                    rollout_ids = seq[prompt_length:].tolist()
                    
                    # Decode to text
                    if self.tokenizer:
                        query_text = self.tokenizer.decode(query_ids, skip_special_tokens=False)
                        rollout_text = self.tokenizer.decode(rollout_ids, skip_special_tokens=False)
                        full_text = self.tokenizer.decode(seq.tolist(), skip_special_tokens=False)
                    else:
                        query_text = str(query_ids)
                        rollout_text = str(rollout_ids)
                        full_text = str(seq.tolist())
                    
                    # Calculate lengths
                    rollout_length = response_mask[i].sum().item() if response_mask is not None else len(rollout_ids)
                    query_length = len(query_ids)
                    
                    # Build record
                    record = {
                        "step": step,
                        "batch_index": i,
                        "query": query_text,
                        "rollout": rollout_text,
                        "full": full_text,
                        "answer": ground_truths[i] if i < len(ground_truths) else None,
                        "reward": float(rewards[i]) if i < len(rewards) and np.isfinite(rewards[i]) else None,
                        "rollout_length": rollout_length,
                        "query_length": query_length,
                        "total_length": len(seq.tolist()),
                    }
                    
                    # Add optional metadata
                    if i < len(record_ids) and record_ids[i]:
                        record["record_id"] = str(record_ids[i])
                    
                    if i < len(origins) and origins[i]:
                        record["origin"] = str(origins[i])
                    
                    if i < len(uids) and uids[i]:
                        record["uid"] = str(uids[i])
                    
                    records.append(record)
                    
                except Exception as e:
                    print(f"Failed to process rollout record {i} at step {step}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            if not records:
                print(f"No valid rollout records to save at step {step}")
                return
            
            # === Sort by rollout length (shortest to longest) ===
            records.sort(key=lambda x: x["rollout_length"])
            
            # === Write to JSONL file ===
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # === Log summary statistics ===
            rollout_lengths = [r["rollout_length"] for r in records]
            query_lengths = [r["query_length"] for r in records]
            valid_rewards = [r["reward"] for r in records if r["reward"] is not None]
            
            logger.info(f"[ROLLOUT-SAVE] Saved {len(records)} rollout records to {output_file}")
            logger.info(f"  Query length: min={min(query_lengths)}, max={max(query_lengths)}, "
                    f"mean={np.mean(query_lengths):.1f}")
            logger.info(f"  Rollout length: min={min(rollout_lengths)}, max={max(rollout_lengths)}, "
                    f"mean={np.mean(rollout_lengths):.1f}, median={np.median(rollout_lengths):.1f}")
            
            if valid_rewards:
                print(f"  Reward stats: min={min(valid_rewards):.3f}, max={max(valid_rewards):.3f}, "
                        f"mean={np.mean(valid_rewards):.3f}")
        except Exception as e:
            # Never crash training due to debug logging
            print(f"Failed to save rollout records at step {step}: {e}")
            import traceback
            traceback.print_exc()
            
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
        """
        Right-pad FULL-SEQUENCE tensors to align batch concatenation.
        Response-only tensors are NOT padded (different natural length).
        """
        # Define which tensors are full-sequence vs response-only
        FULL_SEQUENCE_KEYS = {
            "attention_mask", "sequences", "input_ids", 
            "prompts", "position_ids",
        }
        
        RESPONSE_ONLY_KEYS = {
            "token_level_rewards", "token_level_scores",
            "response_mask", "responses",
            "old_log_probs", "log_probs", "ref_log_probs",
            "advantages", "returns", "values",
            "entropys", "approx_kls",
        }
        
        # Find max length from FULL-SEQUENCE tensors ONLY
        max_length = 0
        for dp in parts:
            for k in FULL_SEQUENCE_KEYS:  # Only check full-sequence tensors
                if k in dp.batch:
                    v = dp.batch[k]
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        max_length = max(max_length, int(v.size(-1)))
        
        target_L = max_length
        
        # Only pad full-sequence tensors
        for dp in parts:
            for k, v in list(dp.batch.items()):
                if not isinstance(v, torch.Tensor) or v.dim() < 2:
                    continue
                
                # SKIP response-only tensors - they stay at original length!
                if k in RESPONSE_ONLY_KEYS:
                    continue
                
                # Only process full-sequence tensors
                if k not in FULL_SEQUENCE_KEYS:
                    continue
                
                # Pad if needed
                curL = int(v.size(-1))
                if curL < target_L:
                    pad = torch.zeros(*v.shape[:-1], target_L - curL,
                                    dtype=v.dtype, device=v.device)
                    dp.batch[k] = torch.cat([v, pad], dim=-1)
        
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
            s = re.sub(r"^```(?:\w+)?\s*|\s*```\$",
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
        # Use token IDs, not text
        token_ids = enc["input_ids"][0].detach().cpu().numpy()

        # Non-tensor fields
        ability = row.get("ability", None)
        data_source = row.get("data_source", "train")
        extra_info = row.get("extra_info", {})
        reward_model = row.get("reward_model", {})

        # Pack into arrays of length 1
        nt = {
            "raw_prompt_data": np.asarray([token_ids], dtype=object),  # TOKEN IDs
            "original_text": np.asarray([text], dtype=object),  # Human-readable
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
    # Lineage tracking methods (NEW)
    # ----------------------------

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

    def _make_array_key(self, arr, idx: int):
        """
        Compact, stable, and format-agnostic key.
        For token arrays: decode to text and hash the text.
        """
        # if idx < 3:
        #     print(f"[KEY-DEBUG] Item {idx}:", flush=True)
        #     print(f"  Type: {type(arr)}", flush=True)
        #     if hasattr(arr, 'shape'):
        #         print(f"  Shape: {arr.shape}, dtype: {arr.dtype if hasattr(arr, 'dtype') else 'N/A'}", flush=True)
        
        # Primary path: Decode tokens to text and hash
        try:
            if hasattr(arr, 'shape') and hasattr(self, 'tokenizer') and self.tokenizer is not None:
                arr_np = np.asarray(arr)
                
                # Handle object dtype arrays (common after padding)
                if arr_np.dtype == object:
                    try:
                        token_list = [int(x) for x in arr_np.flat]
                    except (ValueError, TypeError):
                        # If conversion fails, fall through to text path
                        raise ValueError("Object array contains non-integers")
                else:
                    # Regular numeric array
                    token_list = arr_np.flatten().astype(int).tolist()
                
                # Decode full token sequence to text
                text = self.tokenizer.decode(token_list, skip_special_tokens=True)
                key = ("txt", self._text_hash(text))
                
                # if idx < 3:
                #     print(f"  Key type: txt (decoded), hash: {key[1][:16]}...", flush=True)
                #     print(f"  Text[:100]: {text[:100]}...", flush=True)
                return key
                
        except Exception as e:
            if idx < 3:
                print(f"  Token decode path failed: {e}", flush=True)
        
        # Fallback 1: Full tuple of integers (memory intensive but accurate)
        try:
            arr_np = np.asarray(arr)
            if arr_np.dtype == object:
                arr_int = np.array([int(x) for x in arr_np.flat], dtype=np.int64)
            else:
                arr_int = arr_np.astype(np.int64)
            
            # Use tuple of all values (not str which truncates)
            key = ("ids", tuple(arr_int.flatten().tolist()))
            # if idx < 3:
            #     print(f"  Key type: ids (full tuple), length: {len(key[1])}", flush=True)
            return key
        except Exception as e:
            if idx < 3:
                print(f"  Integer tuple path failed: {e}", flush=True)
        
        # Fallback 2: Hash of raw bytes (if array is contiguous)
        try:
            arr_np = np.asarray(arr)
            if arr_np.dtype == object:
                # Convert to int array first
                arr_np = np.array([int(x) for x in arr_np.flat], dtype=np.int64)
            
            # Hash the raw bytes of the array
            key = ("bytes", hashlib.sha256(arr_np.tobytes()).hexdigest())
            # if idx < 3:
            #     print(f"  Key type: bytes hash, hash: {key[1][:16]}...", flush=True)
            return key
        except Exception as e:
            if idx < 3:
                print(f"  Bytes hash path failed: {e}", flush=True)
        
        # Last resort fallback
        key = ("obj", self._text_hash(f"{type(arr).__name__}:{id(arr)}"))
        # if idx < 3:
        #     print(f"  Key type: obj (id fallback), hash: {key[1][:16]}...", flush=True)
        return key

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

    # ----------------------------
    # Reward refresh methods (NEW)
    # ----------------------------

    def _snapshot_lineage_and_records(self):
        with self._inbox_lock:
            p2c_src = getattr(self, "_parent_to_children", None) or {}
            parent_to_children = {p: set(cs) for p, cs in p2c_src.items()}
            
            # Use index only (always up-to-date)
            records = dict(getattr(self, "_record_index", None) or {})
        
        return parent_to_children, records

    def _reinsert_all_trained(self):
        """Recycle trained items back into pool. No reward refresh needed for prioritized sampling."""
        import time, gc
        if not self._trained_archive or self.query_pool is None:
            return 0

        # Make snapshot
        items = list(self._trained_archive.values())
        self._rng.shuffle(items)
        
        print(f"\n[ARCHIVE-DEBUG] Before reinsertion:")
        print(f"  Archive size: {len(items)}")
        
        # Clear immediately
        self._trained_archive.clear()
        
        print(f"[dynamic] Cleared archive after copying {len(items)} items for reinsertion")
        
        # Process in batches (no reward refresh needed)
        BATCH_SIZE = 512
        total_added = 0
        
        for batch_start in range(0, len(items), BATCH_SIZE):
            batch_items = items[batch_start:batch_start + BATCH_SIZE]
            to_add = []
            
            for idx, base in enumerate(batch_items):
                # Deep copy
                try:
                    rec = QueryRecord(
                        raw_prompt_data=base.raw_prompt_data.copy() if isinstance(base.raw_prompt_data, np.ndarray) else base.raw_prompt_data,
                        input_ids=base.input_ids.clone() if base.input_ids is not None else None,
                        attention_mask=base.attention_mask.clone() if base.attention_mask is not None else None,
                        position_ids=base.position_ids.clone() if base.position_ids is not None else None,
                        gt=base.gt,
                        reward=base.reward,  # Keep existing reward
                        est_reward=base.est_reward,
                        meta=dict(base.meta or {}),
                        record_id=base.record_id,
                        original_text=base.original_text,
                        augmented_text=base.augmented_text,
                        teacher_response=base.teacher_response,
                        creation_time=base.creation_time,
                    )
                except Exception as e:
                    print(f"[COPY-FAIL] Record copy failed: {e}")
                    continue
                
                # Validate
                if not self._validate_and_fix_record(rec):
                    print(f"[VALIDATION-FAIL] Skipping corrupted record")
                    continue
                
                to_add.append(rec)
            
            if to_add:
                self.query_pool.add_many(to_add)
                total_added += len(to_add)
                
                if batch_start + BATCH_SIZE < len(items):
                    time.sleep(0.1)
        
        print(f"[dynamic] Reinserted {total_added}/{len(items)} trained queries")
        
        if total_added > 5000:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_added

    def _is_record_valid_without_fix(self, rec: QueryRecord) -> bool:
        """Check if record is valid WITHOUT modifying it."""
        try:
            data = rec.raw_prompt_data
            if not isinstance(data, np.ndarray):
                return False
            if data.ndim != 1:
                return False
            if data.dtype == object or not np.issubdtype(data.dtype, np.integer):
                return False
            return True
        except:
            return False

    def _validate_and_fix_record(self, rec: QueryRecord) -> bool:
        """Ensure raw_prompt_data is a proper 1-D token array. FIX if possible."""
        try:
            data = rec.raw_prompt_data
            
            # First check tensors if they exist
            if rec.input_ids is not None:
                if not isinstance(rec.input_ids, torch.Tensor):
                    print(f"Record {rec.record_id}: input_ids is not a tensor")
                    return False
                    
                if rec.input_ids.dim() != 1:
                    print(f"Record {rec.record_id}: input_ids has dim={rec.input_ids.dim()}, expected 1")
                    # Try to fix by squeezing
                    if rec.input_ids.numel() > 0:
                        rec.input_ids = rec.input_ids.squeeze()
                        if rec.input_ids.dim() != 1:
                            return False
                    else:
                        return False
                
                if rec.input_ids.numel() == 0:
                    print(f"Record {rec.record_id}: input_ids is empty")
                    return False
            
            if rec.attention_mask is not None:
                if not isinstance(rec.attention_mask, torch.Tensor):
                    print(f"Record {rec.record_id}: attention_mask is not a tensor")
                    return False
                    
                if rec.attention_mask.dim() != 1:
                    print(f"Record {rec.record_id}: attention_mask has dim={rec.attention_mask.dim()}")
                    # Try to fix
                    if rec.attention_mask.numel() > 0:
                        rec.attention_mask = rec.attention_mask.squeeze()
                        if rec.attention_mask.dim() != 1:
                            return False
                    else:
                        return False
                
                # Check shape consistency
                if rec.input_ids is not None and rec.attention_mask.shape != rec.input_ids.shape:
                    print(f"Record {rec.record_id}: attention_mask shape {rec.attention_mask.shape} != input_ids shape {rec.input_ids.shape}")
                    return False
            
            # Fix 0-d arrays
            if isinstance(data, np.ndarray) and data.ndim == 0:
                scalar = data.item()
                
                # If it's text, try to recover from input_ids or original_text
                if isinstance(scalar, (str, bytes)):
                    print(f"Record {rec.record_id}: 0-d text array detected, attempting recovery")
                    
                    # Strategy 1: Use input_ids if available
                    if rec.input_ids is not None:
                        rec.raw_prompt_data = rec.input_ids.detach().cpu().numpy() if hasattr(rec.input_ids, 'detach') \
                                            else np.array(rec.input_ids, dtype=np.int64)
                        logger.info(f"Record {rec.record_id}: Recovered from input_ids")
                        return True
                    
                    # Strategy 2: Re-tokenize from text
                    text = scalar if isinstance(scalar, str) else scalar.decode('utf-8')
                    if self.tokenizer:
                        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                            max_length=self.config.data.max_prompt_length)
                        rec.raw_prompt_data = tokens["input_ids"][0].cpu().numpy()
                        rec.input_ids = tokens["input_ids"][0]
                        rec.attention_mask = tokens["attention_mask"][0]
                        rec.position_ids = _build_position_ids(rec.attention_mask)
                        logger.info(f"Record {rec.record_id}: Recovered via re-tokenization")
                        return True
                    
                    print(f"Record {rec.record_id}: Cannot recover, no tokenizer available")
                    return False
                
                # Numeric scalar - treat as single-token sequence
                try:
                    rec.raw_prompt_data = np.array([int(scalar)], dtype=np.int64)
                    logger.info(f"Record {rec.record_id}: Fixed 0-d numeric array")
                    return True
                except (ValueError, TypeError):
                    print(f"Record {rec.record_id}: Cannot convert scalar to token")
                    return False
            
            # Fix object dtype arrays (might contain nested 0-d arrays or strings)
            if isinstance(data, np.ndarray) and data.dtype == object:
                # Check if it's actually strings
                try:
                    first = data.flat[0]
                    if isinstance(first, (str, bytes)):
                        print(f"Record {rec.record_id}: Object array contains strings, recovering")
                        
                        # Try from input_ids
                        if rec.input_ids is not None:
                            rec.raw_prompt_data = rec.input_ids.detach().cpu().numpy() if hasattr(rec.input_ids, 'detach') \
                                                else np.array(rec.input_ids, dtype=np.int64)
                            logger.info(f"Record {rec.record_id}: Recovered from input_ids")
                            return True
                        
                        # Re-tokenize
                        text = ' '.join(str(x) for x in data.flat)
                        if self.tokenizer:
                            tokens = self.tokenizer(text, return_tensors="pt", truncation=True,
                                                max_length=self.config.data.max_prompt_length)
                            rec.raw_prompt_data = tokens["input_ids"][0].cpu().numpy()
                            rec.input_ids = tokens["input_ids"][0]
                            rec.attention_mask = tokens["attention_mask"][0]
                            rec.position_ids = _build_position_ids(rec.attention_mask)
                            logger.info(f"Record {rec.record_id}: Recovered via re-tokenization")
                            return True
                        
                        return False
                    
                    # Try to convert to proper int array
                    rec.raw_prompt_data = np.array([int(x) for x in data.flat], dtype=np.int64)
                    logger.info(f"Record {rec.record_id}: Converted object array to int64")
                    return True
                except (ValueError, TypeError) as e:
                    print(f"Record {rec.record_id}: Cannot fix object array: {e}")
                    
                    # Last resort: use input_ids
                    if rec.input_ids is not None:
                        rec.raw_prompt_data = rec.input_ids.detach().cpu().numpy() if hasattr(rec.input_ids, 'detach') \
                                            else np.array(rec.input_ids, dtype=np.int64)
                        logger.info(f"Record {rec.record_id}: Last resort recovery from input_ids")
                        return True
                    
                    return False
            
            # Verify it's proper integer array
            if isinstance(data, np.ndarray):
                if data.ndim != 1:
                    print(f"Record {rec.record_id}: Array has ndim={data.ndim}, squeezing")
                    rec.raw_prompt_data = data.squeeze()
                    if rec.raw_prompt_data.ndim != 1:
                        print(f"Record {rec.record_id}: Still not 1-D after squeeze")
                        return False
                
                if not np.issubdtype(data.dtype, np.integer):
                    print(f"Record {rec.record_id}: Wrong dtype {data.dtype}, converting")
                    try:
                        rec.raw_prompt_data = data.astype(np.int64)
                        return True
                    except:
                        print(f"Record {rec.record_id}: Cannot convert to integer dtype")
                        return False
            return True  # Already valid       
        except Exception as e:
            print(f"Failed to validate/fix record {rec.record_id}: {e}")
            return False
        
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
                    print(
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
                print(f"Error loading seed record: {e}")
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
                try:
                    data = r.raw_prompt_data
                    if not isinstance(data, np.ndarray):
                        print(f"[PROTO-CONVERSION] Record {i} has invalid raw_prompt_data type: {type(data).__name__}")
                        continue
                    if data.ndim != 1:
                        print(f"[PROTO-CONVERSION] Record {i} has wrong ndim: {data.ndim}")
                        continue
                    if data.dtype == object or not np.issubdtype(data.dtype, np.integer):
                        print(f"[PROTO-CONVERSION] Record {i} has wrong dtype: {data.dtype}")
                        continue
                    valid_recs.append(r)
                except Exception as e:
                    print(f"[PROTO-CONVERSION] Record {i} validation failed: {e}")
                    continue
            else:
                print(f"Record {i} missing required tensor fields, skipping")
        if not valid_recs:
            raise ValueError("No valid records to convert to DataProto")

        # Figure out lengths and target length
        lens = [int(r.input_ids.size(-1)) for r in valid_recs]
        max_len_in_batch = max(lens) if lens else 0
        cfg_max = int(getattr(self.config.data,
                      "max_prompt_length", max_len_in_batch or 1))
        target_len = cfg_max

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
        
        # Extract actual prompt token IDs for response mask computation
        original_prompt_ids = []
        for r in valid_recs:
            # Store the ORIGINAL (pre-truncation) prompt as list of ints
            if hasattr(r.raw_prompt_data, 'tolist'):
                original_prompt_ids.append(r.raw_prompt_data.tolist())
            else:
                # If it's already a list/tensor
                original_prompt_ids.append(list(r.input_ids.cpu().numpy()))

        # Non-tensor metadata (use np.nan for missing rewards)
        nt = {
            "raw_prompt_data": np.array([r.raw_prompt_data for r in valid_recs], dtype=object),
            "raw_prompt_ids": np.array(original_prompt_ids, dtype=object),
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
        print(f"\n[CONVERT-FLOW-1] PROTO_TO_QUERY_RECORDS")
        print(f"  Input batch size: {len(dp.batch.get('input_ids', []))}")
        
        recs: List[QueryRecord] = []

        if "input_ids" not in dp.batch:
            print("DataProto missing input_ids")
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
                    "policy_difficulty": (float(policy_diffs[i]) if policy_diffs[i] is not None else 1.0),
                },
            )
            if not self._validate_and_fix_record(record):
                print(f"Skipping corrupted record that couldn't be fixed")
                continue
            recs.append(record)

        print(f"  Created records: {len(recs)}")
        print(f"[CONVERT-FLOW-1] END\n")
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
                print(
                    "Tokenizer not initialized, returning string representation")
                return str(token_seq)
        except Exception as e:
            print(f"Failed to decode tokens: {e}")
            return str(token_seq)

    def _tokenize_texts(self, texts: Union(List[str], List[List[Dict]])) -> Dict[str, torch.Tensor]:
        """Tokenize texts with validation and include position_ids."""
        if not texts:
            raise ValueError("Cannot tokenize empty text list")

        max_len = int(getattr(self.config.data, "max_prompt_length", 2048))
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        self.tokenizer.padding_side = "left" # very important, else set right padding by default and cause response length to 1
        if isinstance(texts[0], str):
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
        elif isinstance(texts[0], list):
            enc = self.tokenizer(
                [self.tokenizer.apply_chat_template(text, add_generation_prompt=False, tokenize=False) for text in texts],
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
        else:
            raise NotImplementedError

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
        import sys
        
        # Ensure immediate output
        sys.stdout.reconfigure(line_buffering=True)
        
        print(f"\n{'='*70}")
        print(f"[AUG-FLOW-1] GENERATE_AUGMENTED_QUERIES START")
        print(f"{'='*70}")
        print(f"  num_per_prompt: {num_per_prompt}")
        print(f"  source_batch size: {len(source_batch.batch.get('input_ids', []))}")
        
        try:
            from tensordict import TensorDict as _TD
        except Exception:
            from torchrl.data import TensorDict as _TD

        def _to_td(_d: Dict[str, _torch.Tensor]) -> _TD:
            # Ensure attention_mask exists before position_ids
            if "attention_mask" in _d and "position_ids" not in _d:
                am = _d["attention_mask"]
                try:
                    _d["position_ids"] = _build_position_ids(am)
                except Exception:
                    _d["position_ids"] = torch.zeros_like(am, dtype=torch.long)
            B = 0
            for v in _d.values():
                if isinstance(v, _torch.Tensor):
                    B = int(v.size(0)) if v.dim() > 0 else 0
                    break
            return _TD(_d, batch_size=[B])

        # Handle disabled/zero case
        if num_per_prompt <= 0:
            # print(f"[AUG-DEBUG] EARLY EXIT: num_per_prompt <= 0", flush=True)
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
            # print(f"[AUG-DEBUG] Step 1: Gathering originals from source_batch", flush=True)
            raw_data_list = list(
                source_batch.non_tensor_batch.get(
                    "raw_prompt_data",
                    source_batch.non_tensor_batch.get("raw_prompt_ids", []),
                )
            )
            # print(f"[AUG-DEBUG] raw_data_list length: {len(raw_data_list)}", flush=True)
            
            base_rewards_arr = list(
                source_batch.non_tensor_batch.get(
                    "driver_reward",
                    _np.full(len(raw_data_list), 0.5),
                )
            )
            # print(f"[AUG-DEBUG] base_rewards_arr length: {len(base_rewards_arr)}", flush=True)

            KeyType = Union[Tuple[str, Tuple[int, ...]], Tuple[str, str]]
            key2idx: Dict[KeyType, int] = {}
            originals: List[_np.ndarray] = []
            original_texts: List[str] = []
            base_rewards_per_original: List[float] = []

            for i, arr in enumerate(raw_data_list):
                key = self._make_array_key(arr, i)
                if key not in key2idx:
                    key2idx[key] = len(originals)
                    arr_np = _np.asarray(arr)
                    originals.append(arr_np)
                    original_texts.append(self._decode_tokens_to_text(arr_np))
                    br = float(base_rewards_arr[i]) if i < len(base_rewards_arr) else 0.5
                    br = float(_np.clip(br, -1.0, 1.0)) if _np.isfinite(br) else 0.5
                    base_rewards_per_original.append(br)

            #print(f"[AUG-DEBUG] Unique originals: {len(originals)}", flush=True)
            #print(f"[AUG-DEBUG] Total keys in dict: {len(key2idx)}", flush=True)
            # print(f"[AUG-DEBUG] Sample key: {list(key2idx.keys())[0] if key2idx else 'None'}", flush=True)

            if len(originals) != len(base_rewards_per_original):
                base_rewards_per_original = [0.5] * len(originals)
                # print(f"[AUG-DEBUG] WARNING: Resetting rewards due to length mismatch", flush=True)

            # Extract parent record_ids
            parent_rids_src = list(source_batch.non_tensor_batch.get("record_ids", []))
            key2parent: Dict[KeyType, Optional[str]] = {}
            for i, arr in enumerate(raw_data_list):
                key = self._make_array_key(arr, i)
                if key not in key2parent:
                    pid = str(parent_rids_src[i]) if i < len(parent_rids_src) else None
                    key2parent[key] = pid

            original_parent_ids: List[Optional[str]] = []
            for i, arr in enumerate(originals):
                key = self._make_array_key(arr, i)
                original_parent_ids.append(key2parent.get(key))

            # Use actual per-original rewards when provided
            overrides_hash = None
            if aug_cfg and isinstance(aug_cfg, dict):
                ov = aug_cfg.get("base_reward_overrides_hash")
                if isinstance(ov, dict) and ov:
                    overrides_hash = ov
                    # print(f"[AUG-DEBUG] Using reward overrides: {len(overrides_hash)} entries", flush=True)
                    
            if overrides_hash:
                override_count = 0
                for idx, txt in enumerate(original_texts):
                    key = self._text_hash(txt)
                    if key in overrides_hash:
                        base_rewards_per_original[idx] = float(
                            np.clip(overrides_hash[key], -1.0, 1.0))
                        override_count += 1
                # print(f"[AUG-DEBUG] Applied {override_count} reward overrides", flush=True)

            # Template construction
            # print(f"[AUG-DEBUG] Step 2: Building augmentation prompts", flush=True)
            
            # ==== Anchors (non-codey, plain-text) ====
            ORIG_TAG = "<ORIGINAL>"
            NEW_TAG = "<NEW>"
            DIFF_TAG = "<DIFFICULTY>"
            END_TAG = "<END>"

            def _escape_braces(s: str) -> str:
                return s.replace("{", "{{").replace("}", "}}")

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

            # Example
            ex3_original = (
                r"In \\$\triangle ABC\\$, we have \\$AC=BC=7\\$ and \\$AB=2\\$. Suppose that \\$D\\$ is a point "
                r"on the line \\$AB\\$ such that \\$B\\$ lies between \\$A\\$ and \\$D\\$, and \\$CD=8\\$. "
                r"What is the length of the segment \\$BD\\$?"
            )

            ex3_new = (
                r"In \\$\triangle ABC\\$, we have \\$AC=BC=8\\$ and \\$AB=3\\$. Suppose that \\$D\\$ is a point "
                r"on the line \\$AB\\$ such that \\$B\\$ lies between \\$A\\$ and \\$D\\$, and \\$CD=9\\$. "
                r"What is the length of the segment \\$BD\\$?"
            )
            ex3_diff = 1.03

            parts += [
                f"{ORIG_TAG}\n", _escape_braces(ex3_original) + "\n",
                f"{NEW_TAG}\n",  _escape_braces(ex3_new) + "\n",
                f"{DIFF_TAG}\n", str(ex3_diff) + "\n",
                f"{END_TAG}\n",
            ]

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
                orig_text = original_texts[idx] if idx < len(original_texts) else ""
                
                # Clean the text
                if "is the answer to the problem." in orig_text:
                    orig_text = orig_text.split("is the answer to the problem.")[1]
                if "Remember to put your answer" in orig_text:
                    orig_text = orig_text.split("Remember to put your answer")[0]
                
                cleaned_text = orig_text.strip()
                # print(f"[AUG-DEBUG] Original {idx}: {cleaned_text[:100]}... (reward={br:.3f})", flush=True)
                
                for rep in range(num_per_prompt):
                    aug_prompts.append(TEMPLATE.format(original_problem=cleaned_text))
                    base_rewards_for_aug.append(br)
                    original_texts_for_aug.append(cleaned_text)
                    original_indices.append(idx)
                    parent_ids_for_aug.append(
                        original_parent_ids[idx] if idx < len(original_parent_ids) else None)

            # print(f"[AUG-DEBUG] Total aug_prompts created: {len(aug_prompts)}", flush=True)
            
            if not aug_prompts:
                # print(f"[AUG-DEBUG] EARLY EXIT: No aug_prompts created", flush=True)
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

            # Tokenization
            # print(f"[AUG-DEBUG] Step 3: Tokenizing {len(aug_prompts)} prompts", flush=True)
            enc = self._tokenize_texts(aug_prompts)
            # print(f"[AUG-DEBUG] Tokenized input_ids shape: {enc['input_ids'].shape}", flush=True)
            # print(f"[AUG-DEBUG] Tokenized attention_mask shape: {enc['attention_mask'].shape}", flush=True)
            
            aug_td = _to_td({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "position_ids": enc["position_ids"]
            })
            aug_batch = DataProto(
                batch=aug_td,
                non_tensor_batch={
                    "aug_prompt_len": enc["attention_mask"].sum(dim=1).cpu().numpy()
                },
                meta_info={},
            )

            if "position_ids" not in aug_batch.batch and "attention_mask" in aug_batch.batch:
                aug_batch.batch["position_ids"] = _build_position_ids(
                    aug_batch.batch["attention_mask"])

            # Ensure divisibility by dp_size
            # print(f"[AUG-DEBUG] Step 4: Checking dp_size divisibility", flush=True)
            dp_size = (
                getattr(self.actor_rollout_wg, "world_size", None)
                or getattr(self.actor_rollout_wg, "size", None)
                or getattr(self.actor_rollout_wg, "n_workers", None)
                or getattr(self.actor_rollout_wg, "num_workers", None)
                or 1
            )
            # print(f"[AUG-DEBUG] dp_size={dp_size}", flush=True)
            
            if dp_size > 1:
                try:
                    N = len(aug_batch.batch["input_ids"])
                    # print(f"[AUG-DEBUG] Batch size before alignment: {N}", flush=True)
                    rem = N % dp_size
                    # print(f"[AUG-DEBUG] Remainder: {rem}", flush=True)
                    
                    if rem != 0:
                        keep = N - rem
                        # print(f"[AUG-DEBUG] Trimming to {keep} items for divisibility", flush=True)
                        
                        if keep <= 0:
                            # print(f"[AUG-DEBUG] EARLY EXIT: keep={keep} <= 0 after dp_size alignment", flush=True)
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
                        # print(f"[AUG-DEBUG] Batch size after trim: {len(aug_batch.batch['input_ids'])}", flush=True)
                except Exception as e:
                    # print(f"[AUG-DEBUG] ERROR during dp_size alignment: {e}", flush=True)
                    import traceback
                    traceback.print_exc()

            # Generation call
            # print(f"[AUG-DEBUG] Step 5: Calling generate_sequences", flush=True)
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
                "max_new_tokens": 400,
            })

            try:
                gen_out = self.actor_rollout_wg.generate_sequences(aug_batch)
                # print(f"[AUG-DEBUG] Generation complete", flush=True)
            except Exception as e:
                # print(f"[AUG-DEBUG] ERROR during generate_sequences: {e}", flush=True)
                import traceback
                traceback.print_exc()
                raise

            seq_tensor = gen_out.batch.get("sequences", gen_out.batch.get("input_ids", None))
            if seq_tensor is None:
                # print(f"[AUG-DEBUG] ERROR: generate_sequences did not return 'sequences' or 'input_ids'", flush=True)
                # print(f"[AUG-DEBUG] Available keys in gen_out.batch: {list(gen_out.batch.keys())}", flush=True)
                raise RuntimeError("generate_sequences did not return 'sequences' or 'input_ids'.")

            # print(f"[AUG-DEBUG] Generated sequences shape: {seq_tensor.shape}", flush=True)

            # Decode & parse sections
            # print(f"[AUG-DEBUG] Step 6: Decoding and parsing generations", flush=True)
            prompt_lens = aug_batch.non_tensor_batch["aug_prompt_len"]
            new_texts: List[str] = []
            diff_texts: List[str] = []
            full_generations: List[str] = []

            def _extract_sections(gen_txt: str) -> Tuple[str, str]:
                t = (gen_txt or "").strip()
                if t.startswith("```") and t.endswith("```"):
                    t = re.sub(r"^```(?:\w+)?\s*|\s*```\$", "", t, flags=re.DOTALL).strip()
                t = t.replace("\r\n", "\n")

                def _first_span(text, start_pat, end_pat):
                    m_start = re.search(start_pat, text, flags=re.IGNORECASE)
                    if not m_start:
                        return None, None
                    start = m_start.end()
                    m_end = re.search(end_pat, text[start:], flags=re.IGNORECASE)
                    end = (start + m_end.start()) if m_end else len(text)
                    return start, end

                # Try new tags
                s_new, e_new = _first_span(
                    t, r"<\s*NEW\s*>", r"<\s*(?:DIFFICULTY|END|ORIGINAL|NEW)\s*>")
                if s_new is not None:
                    new_problem = t[s_new:e_new].strip()

                    s_diff, e_diff = _first_span(
                        t, r"<\s*DIFFICULTY\s*>", r"<\s*(?:END|NEW|ORIGINAL)\s*>")
                    diff_src = t[s_diff:e_diff].strip() if s_diff is not None else t[e_new:].strip()
                    return new_problem or t, diff_src or t

                # Fallback: legacy headers
                m_new = re.search(r"#\s*New\s*Problem\s*#", t, flags=re.IGNORECASE)
                if m_new:
                    start = m_new.end()
                    m_next = re.search(
                        r"#\s*(?:Difficulty|End|Original\s*Problem|New\s*Problem)\s*#",
                        t[start:], flags=re.IGNORECASE)
                    end = (start + m_next.start()) if m_next else len(t)
                    new_problem = t[start:end].strip()

                    m_diff = re.search(
                        r"#\s*Difficulty\s*#(?P<body>.*?)(?=#\s*(?:End|New\s*Problem|Original\s*Problem)\s*#|$)",
                        t[end:], flags=re.IGNORECASE | re.DOTALL
                    )
                    diff_src = (m_diff.group("body").strip() if m_diff else t[end:].strip())
                    return new_problem or t, diff_src or t

                return t, t

            for i in range(seq_tensor.size(0)):
                seq_ids = seq_tensor[i].detach().cpu().tolist()
                p_len = int(prompt_lens[i]) if i < len(prompt_lens) else 0
                gen_ids = seq_ids[p_len:] if p_len < len(seq_ids) else []
                txt = self.tokenizer.decode(gen_ids, skip_special_tokens=True) if self.tokenizer else str(gen_ids)
                full_generations.append(txt)
                
                new_prob, diff_src = _extract_sections(txt)
                new_texts.append(new_prob if new_prob else txt)
                diff_texts.append(diff_src)
                
            # print(f"[AUG-DEBUG] Parsed {len(new_texts)} new problems", flush=True)

            # Estimate rewards
            # print(f"[AUG-DEBUG] Step 7: Computing rewards", flush=True)
            # No difficulty-based reward estimation needed for prioritized sampling
            # Just use parent reward as a rough estimate (teacher will validate anyway)
            est_rewards: List[float] = []
            successful_augmentations = []
            
            min_len = min(len(new_texts), len(base_rewards_for_aug), len(parent_ids_for_aug))
            print(f"[AUG-DEBUG] min_len={min_len} for reward estimation")

            for i in range(min_len):
                # Use parent reward as estimate (or default to 0.0 if unknown)
                br = float(base_rewards_for_aug[i]) if i < len(base_rewards_for_aug) else 0.0
                est_rewards.append(br)  # Direct inheritance, no difficulty adjustment
                
                rec = {
                    "record_id": str(uuid.uuid4()),
                    "original_text": original_texts_for_aug[i] if i < len(original_texts_for_aug) else "",
                    "original_index": original_indices[i] if i < len(original_indices) else 0,
                    "augmented_text": new_texts[i],
                    "full_generation": full_generations[i] if i < len(full_generations) else "",
                    "base_reward": br,
                    "estimated_reward": br,  # Same as parent
                    "generation_template": "single_pass",
                    "global_step": getattr(self, "global_steps", 0),
                    "epoch": getattr(self, "current_epoch", 0),
                    # Note: difficulty no longer used for sampling
                }
                if self.augmentation_logger:
                    self.augmentation_logger.log_augmentation(rec)
                self.augmentation_history.append(rec)
                if len(self.augmentation_history) > self.max_history_size:
                    self.augmentation_history.pop(0)
                successful_augmentations.append(rec)

            # Keep metrics tracking for logging purposes
            self._augmentation_metrics["total_augmented"] += len(new_texts[:min_len])
            self._augmentation_metrics["augmentation_success_rate"] = (
                min_len / len(aug_prompts)) if aug_prompts else 0.0

            # Tokenize NEW problems
            # print(f"[AUG-DEBUG] Step 8: Tokenizing {min_len} new problems", flush=True)
            new_enc = self._tokenize_texts(new_texts[:min_len])
            # print(f"[AUG-DEBUG] Final tokenized shape: {new_enc['input_ids'].shape}", flush=True)
            
            # Extract token IDs, not text
            token_ids_list = [new_enc["input_ids"][i].detach().cpu().numpy() 
                      for i in range(min_len)]
            
            record_ids = [r["record_id"] for r in successful_augmentations]
            nt = {
                "raw_prompt_data": np.array(token_ids_list, dtype=object),  # TOKEN IDs
                "augmented_text_readable": np.array(new_texts[:min_len], dtype=object),  # Human-readable
                "policy/est_reward": np.array(est_rewards, dtype=float),
                "original_text": np.array(original_texts_for_aug[:min_len], dtype=object),
                "record_ids": np.array(record_ids, dtype=object),
                "parent_record_id": np.array(parent_ids_for_aug[:min_len], dtype=object),
                "data_source": np.array(["math_dapo"] * min_len, dtype=object),
                "origin": np.array(["augmented"] * min_len, dtype=object),
                "is_augmented": np.array([True] * min_len, dtype=bool),
            }
            new_td = _to_td({
                "input_ids": new_enc["input_ids"],
                "attention_mask": new_enc["attention_mask"],
                "position_ids": new_enc["position_ids"]
            })

            result = DataProto(
                batch=new_td,
                non_tensor_batch=nt,
                meta_info={"augmentation_time": time.time() - gen_start},
            )
            
            print(f"\n[AUG-FLOW-1] GENERATE_AUGMENTED_QUERIES END")
            print(f"  Created {min_len} augmented items")
            print(f"  Token IDs shape: {result.batch['input_ids'].shape}")
            print(f"  Estimated rewards: min={np.min(est_rewards):.3f}, max={np.max(est_rewards):.3f}, mean={np.mean(est_rewards):.3f}")
            print(f"{'='*70}\n")
            
            return result

        except Exception as e:
            print(f"[AUG-DEBUG] ========== EXCEPTION in generate_augmented_queries ==========", flush=True)
            print(f"[AUG-DEBUG] Error type: {type(e).__name__}", flush=True)
            print(f"[AUG-DEBUG] Error message: {e}", flush=True)
            import traceback
            traceback.print_exc()
            print(f"[AUG-DEBUG] ========== End exception trace ==========\n", flush=True)
            
            print(f"Error in generate_augmented_queries: {e}")
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
        """Main thread: integrate all pending teacher-annotated protos into the pool with comprehensive logging."""
        
        # === STAGE 1: DRAIN INBOX ===
        print(f"\n{'='*70}")
        print(f"[INBOX-DRAIN] START - Draining teacher inbox")
        print(f"{'='*70}")
        
        packets: List[DataProto] = []
        with self._inbox_lock:
            inbox_size_before = len(self._annotated_inbox)
            while self._annotated_inbox:
                packets.append(self._annotated_inbox.popleft())
            inbox_size_after = len(self._annotated_inbox)
        
        print(f"[INBOX-DRAIN] Stage 1: Inbox Extraction")
        print(f"  Inbox size before: {inbox_size_before}")
        print(f"  Packets extracted: {len(packets)}")
        print(f"  Inbox size after: {inbox_size_after}")
        
        if not packets:
            print(f"[INBOX-DRAIN] No packets to process, exiting")
            print(f"{'='*70}\n")
            return

        # === STAGE 2: PROCESS PACKETS ===
        print(f"\n[INBOX-DRAIN] Stage 2: Processing {len(packets)} packets")
        
        total_items_in_packets = sum(len(dp.batch.get("input_ids", [])) for dp in packets)
        print(f"  Total items across all packets: {total_items_in_packets}")
        
        integrated_count = 0
        conversion_failures = 0
        validation_failures = 0
        pool_rejections = 0
        
        packet_details = []
        
        for pkt_idx, dp in enumerate(packets):
            packet_info = {
                "index": pkt_idx,
                "input_size": len(dp.batch.get("input_ids", [])),
                "records_created": 0,
                "records_added_to_pool": 0,
            }
            
            print(f"\n  --- Packet {pkt_idx + 1}/{len(packets)} ---")
            print(f"    Input size: {packet_info['input_size']}")
            
            try:
                # Convert DataProto to QueryRecords
                print(f"    Converting to QueryRecords...")
                new_items: List[QueryRecord] = self._proto_to_query_records(dp)
                packet_info["records_created"] = len(new_items)
                print(f"    Created {len(new_items)} records")
                
                if len(new_items) == 0:
                    conversion_failures += packet_info['input_size']
                    print(f"    ⚠️  No records created (conversion filtered all items)")
                    packet_details.append(packet_info)
                    continue
                
                # Register in lineage & index
                print(f"    Registering in lineage tracking...")
                self._register_records(new_items)
                
                # Ensure rewards are set from est
                ests = dp.non_tensor_batch.get("policy/est_reward", [])
                record_ids = dp.non_tensor_batch.get("record_ids", [])
                est_arr = dp.non_tensor_batch.get("policy/est_reward", [])
                
                rewards_set = 0
                for idx, it in enumerate(new_items):
                    if idx < len(ests):
                        try:
                            reward_val = float(ests[idx])
                            if np.isfinite(reward_val):
                                it.reward = np.clip(reward_val, -1.0, 1.0)
                                it.est_reward = it.reward
                                rewards_set += 1
                        except (TypeError, ValueError):
                            pass
                
                print(f"    Set rewards for {rewards_set}/{len(new_items)} records")
                
                # Validate records before adding
                print(f"    Validating records...")
                valid_items = []
                invalid_count = 0
                
                for it in new_items:
                    if self._validate_and_fix_record(it):
                        valid_items.append(it)
                    else:
                        invalid_count += 1
                        validation_failures += 1
                
                print(f"    Validation: {len(valid_items)} valid, {invalid_count} invalid")
                
                if not valid_items:
                    print(f"    ⚠️  No valid records to add")
                    packet_details.append(packet_info)
                    continue
                
                # Add to pool
                print(f"    Adding to pool...")
                pool_size_before = self.query_pool.size()
                
                self.query_pool.add_many(valid_items)
                
                pool_size_after = self.query_pool.size()
                actual_added = pool_size_after - pool_size_before
                packet_info["records_added_to_pool"] = actual_added
                
                print(f"    Pool: {pool_size_before} → {pool_size_after} (Δ{actual_added:+d})")
                
                integrated_count += actual_added
                pool_rejections += len(valid_items) - actual_added
                
                # Log detailed info if discrepancy
                if actual_added < len(valid_items):
                    print(f"    ⚠️  Pool accepted {actual_added}/{len(valid_items)} valid records")
                    print(f"       {len(valid_items) - actual_added} records were rejected/evicted by pool")
                
                packet_details.append(packet_info)
                
            except Exception as e:
                print(f"    ❌ ERROR processing packet {pkt_idx}: {e}")
                import traceback
                traceback.print_exc()
                conversion_failures += packet_info['input_size']
                packet_details.append(packet_info)
                
                # Log integration failure
                if self.augmentation_logger:
                    self.augmentation_logger.log_annotation({
                        "event": "integration_failed",
                        "error": str(e),
                        "packet_index": pkt_idx,
                        "batch_size": packet_info['input_size'],
                        "global_step": getattr(self, 'global_steps', 0),
                    })

        # === STAGE 3: UPDATE METRICS ===
        print(f"\n[INBOX-DRAIN] Stage 3: Updating Metrics")
        
        self._augmentation_metrics["total_teacher_integrated"] += integrated_count
        
        # Calculate success rates
        success_rate = (integrated_count / total_items_in_packets * 100) if total_items_in_packets > 0 else 0
        
        print(f"  Total items processed: {total_items_in_packets}")
        print(f"  Successfully integrated: {integrated_count} ({success_rate:.1f}%)")
        print(f"  Conversion failures: {conversion_failures}")
        print(f"  Validation failures: {validation_failures}")
        print(f"  Pool rejections: {pool_rejections}")
        
        # === STAGE 4: DETAILED BREAKDOWN ===
        print(f"\n[INBOX-DRAIN] Stage 4: Packet-by-Packet Breakdown")
        
        successful_packets = sum(1 for p in packet_details if p["records_added_to_pool"] > 0)
        failed_packets = len(packet_details) - successful_packets
        
        print(f"  Successful packets: {successful_packets}/{len(packets)}")
        print(f"  Failed packets: {failed_packets}/{len(packets)}")
        
        # Show details of failed packets
        if failed_packets > 0:
            print(f"\n  Failed Packet Details:")
            for p in packet_details:
                if p["records_added_to_pool"] == 0:
                    print(f"    Packet {p['index']}:")
                    print(f"      Input: {p['input_size']}, Created: {p['records_created']}, Added: {p['records_added_to_pool']}")
        
        # Show summary of successful packets
        if successful_packets > 0:
            print(f"\n  Successful Packet Summary:")
            total_created = sum(p["records_created"] for p in packet_details)
            total_added = sum(p["records_added_to_pool"] for p in packet_details)
            efficiency = (total_added / total_created * 100) if total_created > 0 else 0
            print(f"    Total records created: {total_created}")
            print(f"    Total records added: {total_added}")
            print(f"    Pool acceptance rate: {efficiency:.1f}%")
        
        # === STAGE 5: LOG SAMPLES ===
        if self.augmentation_logger and integrated_count > 0:
            print(f"\n[INBOX-DRAIN] Stage 5: Logging Samples")
            sample_count = min(10, integrated_count)
            print(f"  Logging {sample_count} sample records to augmentation logger")
            
            # Log first successful packet's records
            for p in packet_details:
                if p["records_added_to_pool"] > 0:
                    # Get the corresponding packet
                    pkt_idx = p["index"]
                    if pkt_idx < len(packets):
                        dp = packets[pkt_idx]
                        record_ids = dp.non_tensor_batch.get("record_ids", [])
                        est_rewards = dp.non_tensor_batch.get("policy/est_reward", [])
                        
                        for i in range(min(sample_count, len(record_ids))):
                            self.augmentation_logger.log_annotation({
                                "event": "teacher_batch_integrated",
                                "record_id": str(record_ids[i]) if i < len(record_ids) else None,
                                "estimated_reward": float(est_rewards[i]) if i < len(est_rewards) else None,
                                "packet_index": pkt_idx,
                                "global_step": getattr(self, 'global_steps', 0),
                            })
                    break
        
        # === FINAL SUMMARY ===
        print(f"\n{'='*70}")
        print(f"[INBOX-DRAIN] COMPLETE")
        print(f"{'='*70}")
        print(f"Summary:")
        print(f"  Input: {total_items_in_packets} items in {len(packets)} packets")
        print(f"  Output: {integrated_count} items integrated ({success_rate:.1f}% success)")
        print(f"  Loss Breakdown:")
        print(f"    - Conversion failures: {conversion_failures} ({conversion_failures/total_items_in_packets*100:.1f}%)" if total_items_in_packets > 0 else "    - Conversion failures: 0")
        print(f"    - Validation failures: {validation_failures} ({validation_failures/total_items_in_packets*100:.1f}%)" if total_items_in_packets > 0 else "    - Validation failures: 0")
        print(f"    - Pool rejections: {pool_rejections} ({pool_rejections/total_items_in_packets*100:.1f}%)" if total_items_in_packets > 0 else "    - Pool rejections: 0")
        print(f"{'='*70}\n")

    def _prepare_reward_model_inputs(self, dp: DataProto) -> None:
        """
        Populate dp.non_tensor_batch['reward_model'] as an object array of per-item dicts:
            reward_model[i] -> {'ground_truth': , 'style': , 'solvable': }
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

    def _maybe_top_up_pool_for_next_batch(self, want: int) -> int:
        """If pool has fewer than `want` items, clone enough seeds and add them back."""
        if self.query_pool is None or want <= 0:
            return 0
        curr = self.query_pool.size()
        need = max(0, want - curr)
        if need == 0 or not self._seed_records_template:
            return 0

        # Use reseed_round counter if available
        if not hasattr(self, '_reseed_round'):
            self._reseed_round = 0
        self._reseed_round += 1

        # clone exactly `need` seeds (cycle over the template if smaller than need)
        clones: List[QueryRecord] = []
        S = len(self._seed_records_template)
        for i in range(need):
            base = self._seed_records_template[i % S]
            clones.append(self._clone_seed_record(base, epoch=getattr(
                self, "current_epoch", 0), reseed_round=self._reseed_round))

        self.query_pool.add_many(clones)

        # optional: log a small snapshot
        if self.augmentation_logger:
            self.augmentation_logger.log_pool_snapshot(
                {"event": "top_up_for_batch", "added": len(clones), "want": want, "before": curr,
                 "after": self.query_pool.size(), "reseed_round": self._reseed_round},
                [c.to_dict() for c in clones[:10]]
            )
        return len(clones)

    # ----------------------------
    # Main training loop with comprehensive logging
    # ----------------------------

    def fit(self):
        """Main training loop with checkpoint recovery."""
        from omegaconf import OmegaConf
        from collections import defaultdict
        from verl.utils.tracking import Tracking
        from verl.utils.profiler import marked_timer

        logger_instance = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        # Initialize step counters (may be overwritten by checkpoint)
        self.global_steps = 1
        self.gen_steps = 1
        self.current_epoch = 0

        # Load model checkpoint FIRST (this sets self._last_checkpoint_path)
        self._load_checkpoint()

        # Dynamic data configuration
        dyn_en = bool(getattr(self.config, "dynamic_data", {}).get("enable", False))
        
        # Initialize structures needed for both fresh start AND state loading
        self._inbox_lock = threading.Lock()
        self._annotated_inbox: deque[DataProto] = deque(maxlen=20000)
        self.teacher_annotator: Optional[AsyncTeacherAnnotator] = None
        self._rng = np.random.default_rng()
        
        # Initialize lineage tracking (needed before state load)
        self._ensure_lineage_maps()
        self._trained_archive: Dict[str, QueryRecord] = {}
        self._seed_records_template: List[QueryRecord] = []
        self._reseed_round: int = 0
        
        state_loaded = False
        
        if dyn_en:
            # Try to restore training state from checkpoint (global_steps set by _load_checkpoint)
            if self.global_steps > 1:
                _ckpt_dir = os.path.join(
                    self.config.trainer.default_local_dir,
                    f"global_step_{self.global_steps}"
                )
                logger.info(f"Attempting to restore training state from {_ckpt_dir}...")
                state_loaded = self._load_training_state(_ckpt_dir)
            
            # Only initialize fresh pool if state wasn't loaded
            if not state_loaded:
                logger.info("Initializing fresh query pool and loading seed data...")
                
                # Create fresh pool
                max_pool_size = int(getattr(self.config.dynamic_data, "max_pool_size", 30000))
                epsilon = float(getattr(self.config.dynamic_data, "sampling_epsilon", 0.1))
                temperature = float(getattr(self.config.dynamic_data, "sampling_temperature", 1.0))

                self.query_pool = PrioritizedQueryPool(
                    max_size=max_pool_size,
                    epsilon=epsilon,
                    temperature=temperature,
                    trainer_ref=self
                )

                logger.info(f"Initialized PrioritizedQueryPool with epsilon={epsilon}, temperature={temperature}")
                
                # Load seed records from dataloader
                seed_records = self._seed_records_from_loader()
                init_mode = str(getattr(self.config.dynamic_data, "init_mode", "map")).lower()

                self.query_pool.add_many(seed_records)

                # Keep template for future top-ups
                self._seed_records_template = [
                    self._clone_seed_record(r, epoch=0, reseed_round=0) 
                    for r in seed_records
                ]

                # Register seeds for lineage tracking
                self._register_records(seed_records)
                
                logger.info(f"Fresh pool initialized with {len(seed_records)} seed records")
            else:
                logger.info("✓ Training state restored from checkpoint - skipping fresh initialization")
                # Verify pool was restored correctly
                if not hasattr(self, 'query_pool') or self.query_pool is None:
                    raise RuntimeError("State loading succeeded but query_pool is None!")
                logger.info(f"  Restored pool has {self.query_pool.size()} items")
                logger.info(f"  Restored archive has {len(self._trained_archive)} items")
            
            # Start teacher annotator (needed regardless of state load)
            try:
                model_name = getattr(self.config.dynamic_data, "teacher_model", "gpt-4o-mini")
                self.teacher_annotator = AsyncTeacherAnnotator(
                    self,
                    model_name=model_name,
                    augmentation_logger=self.augmentation_logger,
                    immediate_release=True,
                )
                self.teacher_annotator.start()
                logger.info("✓ Teacher annotator thread started")
            except Exception as e:
                logger.error(f"Failed to start teacher annotator: {e}")
                raise
        else:
            # Dynamic data disabled - no pool needed
            self.query_pool = None
            logger.info("Dynamic data disabled, no query pool created")

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

                    # Drain any teacher results into the pool
                    self._drain_teacher_inbox()

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
                            print("[REINSERT-DEBUG] Reinserted trained queries:", added)
                            logger.info(
                                f"[dynamic] pool<{want}; recycled {added} trained queries")
                        else:
                            print("[REINSERT-DEBUG] Pool size sufficient, no reinsertion.")

                        sampled = self.query_pool.sample_batch(k=want)
                        if sampled:
                            print(f"\n[POST-SAMPLE-VALIDATION] Validating {len(sampled)} sampled records...")
                            invalid_sampled = []
                            for i, rec in enumerate(sampled):
                                data = rec.raw_prompt_data
                                if not isinstance(data, np.ndarray):
                                    invalid_sampled.append((i, f"not ndarray: {type(data).__name__}"))
                                elif data.ndim != 1:
                                    invalid_sampled.append((i, f"wrong ndim: {data.ndim}"))
                                elif data.dtype == object or not np.issubdtype(data.dtype, np.integer):
                                    invalid_sampled.append((i, f"wrong dtype: {data.dtype}"))
                                elif data.size == 0:
                                    invalid_sampled.append((i, "empty array"))
                            
                            if invalid_sampled:
                                print(f"[POST-SAMPLE-VALIDATION] ❌ Found {len(invalid_sampled)} INVALID records in sample!")
                                for idx, reason in invalid_sampled[:5]:  # Show first 5
                                    rec = sampled[idx]
                                    print(f"  Item {idx}: {reason}")
                                    print(f"    record_id: {rec.record_id}")
                                    print(f"    origin: {(rec.meta or {}).get('origin', 'unknown')}")
                                    print(f"    trained_round: {(rec.meta or {}).get('trained_round', 'N/A')}")
                                
                                # CRITICAL: Remove invalid records before proceeding
                                sampled = [rec for i, rec in enumerate(sampled) if i not in [idx for idx, _ in invalid_sampled]]
                                print(f"[POST-SAMPLE-VALIDATION] Removed invalid records, continuing with {len(sampled)} valid items")
                            else:
                                print(f"[POST-SAMPLE-VALIDATION] ✓ All {len(sampled)} records passed validation")
                                
                            rewards_pre_rollout = [r.reward for r in sampled]
                            origins_pre_rollout = [(r.meta or {}).get("origin", "unknown") for r in sampled]
                            
                            from collections import Counter
                            origin_counts = Counter(origins_pre_rollout)
                            
                            # Stats by origin (FILTER OUT None values)
                            origin_reward_stats = {}
                            for origin_type in set(origins_pre_rollout):
                                # Filter None before computing stats
                                origin_rewards = [r for r, o in zip(rewards_pre_rollout, origins_pre_rollout) 
                                                if o == origin_type and r is not None]
                                
                                if origin_rewards:  # Only compute if we have valid rewards
                                    origin_reward_stats[origin_type] = {
                                        "count": len(origin_rewards),
                                        "mean": float(np.mean(origin_rewards)),
                                        "min": float(np.min(origin_rewards)),
                                        "max": float(np.max(origin_rewards)),
                                        "std": float(np.std(origin_rewards)),
                                    }
                                else:
                                    # No valid rewards for this origin
                                    origin_reward_stats[origin_type] = {
                                        "count": len([r for r, o in zip(rewards_pre_rollout, origins_pre_rollout) if o == origin_type]),
                                        "mean": None,
                                        "min": None,
                                        "max": None,
                                        "std": None,
                                        "note": "all_None_rewards",
                                    }
                            
                            # Overall stats (filter None)
                            valid_rewards = [r for r in rewards_pre_rollout if r is not None]
                            
                            print(f"\n[SAMPLE-DEBUG] Step {self.global_steps} sampled {len(sampled)} items:")
                            print(f"  Origin counts: {dict(origin_counts)}")
                            print(f"  Valid rewards: {len(valid_rewards)}/{len(sampled)} (None count: {len(sampled) - len(valid_rewards)})")
                            print(f"  Pre-rollout reward stats by origin:")
                            for origin, stats in origin_reward_stats.items():
                                print(f"    {origin}: {stats}")
                            if valid_rewards:
                                print(f"  Overall pre-rollout reward: mean={np.mean(valid_rewards):.3f}, "
                                    f"std={np.std(valid_rewards):.3f}, min={np.min(valid_rewards):.3f}, max={np.max(valid_rewards):.3f}")
                            else:
                                print(f"  Overall pre-rollout reward: ALL NONE!")
                        if not sampled:
                            print("[dynamic] queue is empty; waiting…")
                            time.sleep(0.1)
                            continue

                        sampled_records_for_this_step = sampled  # keep for reward update

                        
                        print(f"\n[SCORED-ITEM-DEBUG] Validating scored items from pool:")
                        scored_items = [r for r in sampled_records_for_this_step if (r.meta or {}).get("origin") != "seed"]
                        if scored_items:
                            print(f"  Found {len(scored_items)} scored (non-seed) items")
                            issues = []
                            for i, rec in enumerate(scored_items[:5]):  # Check first 5
                                try:
                                    # Check tensors are valid
                                    if rec.input_ids is None:
                                        issues.append(f"Item {i}: input_ids is None")
                                    elif not isinstance(rec.input_ids, torch.Tensor):
                                        issues.append(f"Item {i}: input_ids is not a tensor ({type(rec.input_ids)})")
                                    elif rec.input_ids.numel() == 0:
                                        issues.append(f"Item {i}: input_ids is empty")
                                    
                                    if rec.attention_mask is None:
                                        issues.append(f"Item {i}: attention_mask is None")
                                    elif rec.input_ids is not None and rec.attention_mask.shape != rec.input_ids.shape:
                                        issues.append(f"Item {i}: shape mismatch (ids:{rec.input_ids.shape} vs mask:{rec.attention_mask.shape})")
                                    
                                    # Check raw_prompt_data
                                    if not isinstance(rec.raw_prompt_data, np.ndarray):
                                        issues.append(f"Item {i}: raw_prompt_data is not ndarray")
                                    elif rec.raw_prompt_data.dtype == object:
                                        issues.append(f"Item {i}: raw_prompt_data has object dtype")
                                        
                                except Exception as e:
                                    issues.append(f"Item {i}: validation exception: {e}")
                            
                            if issues:
                                print(f"  ISSUES FOUND:")
                                for issue in issues:
                                    print(f"    {issue}")
                            else:
                                print(f"  ✓ All sampled scored items passed validation")
                        
                        

                        # Take snapshot BEFORE any updates happen
                        pre_rollout_snapshot = {
                            "cached_rewards": [r.reward for r in sampled_records_for_this_step],
                            "est_rewards": [r.est_reward for r in sampled_records_for_this_step],
                            "origins": [(r.meta or {}).get("origin", "unknown") for r in sampled_records_for_this_step],
                            "record_ids": [r.record_id for r in sampled_records_for_this_step],
                        }

                        print(f"\n[PRE-ROLLOUT-SNAPSHOT] Step {self.global_steps}: Captured state BEFORE rollout")
                        valid_cached = sum(1 for r in pre_rollout_snapshot["cached_rewards"] 
                                        if r is not None and np.isfinite(r))
                        valid_est = sum(1 for r in pre_rollout_snapshot["est_rewards"] 
                                        if r is not None and np.isfinite(r))
                        print(f"  Items with cached rewards: {valid_cached}/{len(sampled_records_for_this_step)}")
                        print(f"  Items with estimated rewards: {valid_est}/{len(sampled_records_for_this_step)}")

                        from collections import Counter
                        origin_counts = Counter(pre_rollout_snapshot["origins"])
                        print(f"  Origins: {dict(origin_counts)}")

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
                    _wanted_nt = ("raw_prompt_data", "driver_reward", "record_ids")
                    _nt_to_pop = [
                        k for k in _wanted_nt if k in new_batch.non_tensor_batch]

                    gen_batch = new_batch.pop(
                        batch_keys=pop_keys,
                        non_tensor_batch_keys=_nt_to_pop,
                    )
                    
                    print("[FIT-DEBUG-1] Preparing to generate rollouts...")
                    
                    # ============ ADD TENSOR VALIDATION ============
                    print(f"\n[TENSOR-VALIDATION] Validating {len(gen_batch.batch['input_ids'])} items before generation")

                    corrupted_indices = []
                    device_issues = []
                    shape_issues = []

                    # First pass: identify all issues
                    for i in range(len(gen_batch.batch["input_ids"])):
                        try:
                            # Validate input_ids
                            ids = gen_batch.batch["input_ids"][i]
                            
                            # Check dimension
                            if ids.dim() != 1:
                                shape_issues.append(i)
                                print(f"[TENSOR-VALIDATION] Item {i}: input_ids wrong dim {ids.dim()}, expected 1")
                                corrupted_indices.append(i)
                                continue
                            
                            # Check not empty
                            if ids.numel() == 0:
                                shape_issues.append(i)
                                print(f"[TENSOR-VALIDATION] Item {i}: input_ids is empty")
                                corrupted_indices.append(i)
                                continue
                            
                            # Validate attention_mask
                            mask = gen_batch.batch["attention_mask"][i]
                            if mask.dim() != 1:
                                shape_issues.append(i)
                                print(f"[TENSOR-VALIDATION] Item {i}: attention_mask wrong dim {mask.dim()}")
                                corrupted_indices.append(i)
                                continue
                                
                            if mask.shape != ids.shape:
                                shape_issues.append(i)
                                print(f"[TENSOR-VALIDATION] Item {i}: attention_mask shape {mask.shape} != input_ids shape {ids.shape}")
                                corrupted_indices.append(i)
                                continue
                            
                            # Validate position_ids if present
                            if "position_ids" in gen_batch.batch:
                                pos = gen_batch.batch["position_ids"][i]
                                if pos.dim() != 1:
                                    shape_issues.append(i)
                                    print(f"[TENSOR-VALIDATION] Item {i}: position_ids wrong dim {pos.dim()}")
                                    corrupted_indices.append(i)
                                    continue
                                    
                                if pos.shape != ids.shape:
                                    shape_issues.append(i)
                                    print(f"[TENSOR-VALIDATION] Item {i}: position_ids shape {pos.shape} != input_ids shape {ids.shape}")
                                    corrupted_indices.append(i)
                                    continue
                            
                            # Check device consistency (first item sets expected device)
                            if i == 0:
                                expected_device = ids.device
                                print(f"[TENSOR-VALIDATION] Expected device: {expected_device}")
                            else:
                                if ids.device != expected_device:
                                    device_issues.append(i)
                                    print(f"[TENSOR-VALIDATION] Item {i}: device {ids.device} != expected {expected_device}")
                                    corrupted_indices.append(i)
                                    continue
                                
                                if mask.device != expected_device:
                                    device_issues.append(i)
                                    print(f"[TENSOR-VALIDATION] Item {i}: attention_mask device {mask.device} != expected {expected_device}")
                                    corrupted_indices.append(i)
                                    continue
                            
                        except Exception as e:
                            corrupted_indices.append(i)
                            print(f"[TENSOR-VALIDATION] Item {i}: exception during validation: {e}")
                            import traceback
                            traceback.print_exc()

                    # Summary of issues
                    if corrupted_indices:
                        print(f"\n[TENSOR-VALIDATION] ❌ Found {len(corrupted_indices)} corrupted items:")
                        print(f"  - Shape issues: {len(shape_issues)}")
                        print(f"  - Device issues: {len(device_issues)}")
                        
                        # Remove corrupted items
                        keep_indices = [i for i in range(len(gen_batch.batch["input_ids"])) if i not in corrupted_indices]
                        
                        if not keep_indices:
                            print(f"[TENSOR-VALIDATION] CRITICAL: All items corrupted, skipping this step")
                            continue
                        
                        print(f"[TENSOR-VALIDATION] Keeping {len(keep_indices)}/{len(gen_batch.batch['input_ids'])} valid items")
                        gen_batch = gen_batch[keep_indices]
                        
                    else:
                        print(f"[TENSOR-VALIDATION] ✓ All {len(gen_batch.batch['input_ids'])} items passed validation")

                    # Ensure divisibility by dp_size
                    dp_size = (
                        getattr(self.actor_rollout_wg, "world_size", None)
                        or getattr(self.actor_rollout_wg, "size", None)
                        or getattr(self.actor_rollout_wg, "n_workers", None)
                        or 1
                    )

                    batch_size = len(gen_batch.batch["input_ids"])
                    print(f"[TENSOR-VALIDATION] Checking dp_size divisibility: batch_size={batch_size}, dp_size={dp_size}")

                    if dp_size > 1 and batch_size % dp_size != 0:
                        keep = (batch_size // dp_size) * dp_size
                        print(f"[TENSOR-VALIDATION] ⚠ Trimming {batch_size} -> {keep} for dp_size={dp_size} divisibility")
                        if keep > 0:
                            gen_batch = gen_batch[:keep]
                            if 'gen_batch_unrepeated' in locals():
                                gen_batch_unrepeated = gen_batch_unrepeated[:keep]
                        else:
                            print(f"[TENSOR-VALIDATION] CRITICAL: Cannot satisfy dp_size requirement, skipping step")
                            continue
                    else:
                        print(f"[TENSOR-VALIDATION] ✓ Batch size {batch_size} is divisible by dp_size {dp_size}")

                    print(f"[TENSOR-VALIDATION] ✓ Final validation complete: {len(gen_batch.batch['input_ids'])} items ready\n")
                    # ============ END TENSOR VALIDATION ============
                    
                    # ===== IMPORTANT: Save unrepeated batch for augmentation =====
                    gen_batch_unrepeated = gen_batch  # Keep reference to original

                    # ===== DEBUG: Sequence length analysis =====
                    print(f"\n[SEQUENCE-LENGTH-DEBUG] Analyzing batch composition:")
                    try:
                        origins = list(gen_batch.non_tensor_batch.get("origin", ["unknown"] * len(gen_batch.batch["input_ids"])))
                        seq_lens = gen_batch.batch["attention_mask"].sum(dim=-1).cpu().numpy()
                        
                        from collections import defaultdict
                        origin_lens = defaultdict(list)
                        for origin, length in zip(origins, seq_lens):
                            origin_lens[origin].append(int(length))
                        
                        print(f"  Sequence lengths by origin:")
                        for origin, lengths in origin_lens.items():
                            print(f"    {origin}: count={len(lengths)}, mean={np.mean(lengths):.1f}, "
                                f"min={np.min(lengths)}, max={np.max(lengths)}, "
                                f"total_tokens={np.sum(lengths)}")
                        
                        total_prompt_tokens = int(np.sum(seq_lens))
                        max_gen_tokens = int(gen_batch.meta_info.get("max_new_tokens", 2048))
                        estimated_total = total_prompt_tokens + (len(seq_lens) * max_gen_tokens)
                        print(f"  Total prompt tokens: {total_prompt_tokens:,}")
                        print(f"  Estimated after generation: {estimated_total:,} "
                            f"({estimated_total / 1e9:.2f}B tokens)")
                        
                    except Exception as e:
                        print(f"  ERROR in sequence length analysis: {e}")
                    # ===== End sequence length analysis =====
                    print(f"\n[TENSOR-DEVICE-DEBUG] Checking tensor properties:")
                    try:
                        for key in ["input_ids", "attention_mask", "position_ids"]:
                            if key in gen_batch.batch:
                                tensor = gen_batch.batch[key]
                                print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}, "
                                    f"device={tensor.device}, contiguous={tensor.is_contiguous()}")
                                
                                # Check for invalid values
                                if key == "input_ids":
                                    if torch.any(tensor < 0):
                                        print(f"    WARNING: Negative token IDs found!")
                                    if torch.any(tensor > 100000):  # Reasonable vocab size
                                        print(f"    WARNING: Suspiciously large token IDs found!")
                        
                        # Check position_ids specifically
                        if "position_ids" in gen_batch.batch:
                            pos = gen_batch.batch["position_ids"]
                            if torch.any(pos < 0):
                                print(f"  WARNING: Negative position_ids found!")
                            if torch.any(pos > 50000):  # Reasonable max position
                                print(f"  WARNING: Suspiciously large position_ids found!")
                                
                    except Exception as e:
                        print(f"  ERROR in tensor device check: {e}")
                        import traceback
                        traceback.print_exc()
                    # ===== End tensor/device debug =====


                    # Repeat gen_batch for rollout
                    gen_batch = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True
                    )

                    if "position_ids" not in gen_batch.batch and "attention_mask" in gen_batch.batch:
                        gen_batch.batch["position_ids"] = _build_position_ids(
                            gen_batch.batch["attention_mask"])
                        # Also fix position_ids for unrepeated batch
                        if "position_ids" not in gen_batch_unrepeated.batch:
                            gen_batch_unrepeated.batch["position_ids"] = _build_position_ids(
                                gen_batch_unrepeated.batch["attention_mask"])


                    print(f"\n[MEMORY-DEBUG] Pre-generation memory state:")
                    try:
                        import psutil
                        process = psutil.Process()
                        mem_info = process.memory_info()
                        print(f"  CPU memory: {mem_info.rss / 1024**3:.2f} GB")
                        
                        if torch.cuda.is_available():
                            for i in range(torch.cuda.device_count()):
                                print(f"  GPU {i} allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                                print(f"  GPU {i} reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
                                print(f"  GPU {i} free: {(torch.cuda.mem_get_info(i)[0]) / 1024**3:.2f} GB")
                    except Exception as e:
                        print(f"  ERROR checking memory: {e}")

                    print(f"[TRAIN-DEBUG] Global step {self.global_steps}: gen_batch size {len(gen_batch.batch.get('input_ids', []))}")
                    print("[FIT-DEBUG-2] Start generating rollouts...")
                    with marked_timer("step", timing_raw):
                        # Step 3: generate rollouts
                        with marked_timer("gen", timing_raw, "red"):
                            self._dump_debug_batch(gen_batch_unrepeated, self.global_steps)
                            print("[FIT-DEBUG] Dumped debug batch before generation.")
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(
                                gen_batch)
                            timing_raw.update(
                                gen_batch_output.meta_info.get("timing", {}))
                            gen_batch_output.meta_info.pop("timing", None)

                        print("[FIT-DEBUG-3] Rollout initial generation completed. Start baseline calculation...")

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

                        print("[FIT-DEBUG-4] Baseline calculation completed. Preparing for reward calculation...")

                        # Correct UID creation
                        n_rep = int(self.config.actor_rollout_ref.rollout.n)
                        gen_bsz = int(gen_batch.batch["input_ids"].size(0))
                        if gen_bsz % n_rep != 0:
                            print(
                                f"Repeated size {gen_bsz} not divisible by n_rep={n_rep}")
                            continue

                        base_bsz = gen_bsz // n_rep
                        new_batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(base_bsz)], dtype=object
                        )

                        print("[FIT-DEBUG-5] Preparing repeated batch for reward calculation...")

                        new_batch = new_batch.repeat(
                            repeat_times=n_rep, interleave=True)
                        new_batch = new_batch.union(gen_batch_output)

                        print("[FIT-DEBUG-6] Generation completed. Start reward calculation...")

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
                                print(
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

                            dump_path = getattr(
                                self.config.dynamic_data, "origin_reward_dump_path", "./logs/origin_rewards.jsonl")
                            self._dump_origin_rewards(
                                new_batch, dump_path, self.global_steps)

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

                                # 1. First update success rates (this promotes cold items)
                                success_rate_updates = []
                                for i, rec in enumerate(sampled_records_for_this_step):
                                    if i >= len(base_uid_order):
                                        break
                                    u = base_uid_order[i]
                                    if u not in uid2rewards:
                                        continue
                                    avg_r = float(np.mean(uid2rewards[u]))
                                    success_rate_updates.append((rec, avg_r))

                                if success_rate_updates:
                                    self.query_pool.update_success_rates(success_rate_updates)
                                    print(f"[SUCCESS-RATE-UPDATE] Updated {len(success_rate_updates)} records")

                                # 2. THEN archive (now all items are in _records)
                                for i, rec in enumerate(sampled_records_for_this_step):
                                    if i >= len(base_uid_order):
                                        break
                                    u = base_uid_order[i]
                                    if u not in uid2rewards:
                                        continue
                                    avg_r = float(np.mean(uid2rewards[u]))
                                    
                                    # Create archived copy
                                    archived_rec = QueryRecord(
                                        raw_prompt_data=rec.raw_prompt_data.copy() if isinstance(rec.raw_prompt_data, np.ndarray) else rec.raw_prompt_data,
                                        input_ids=rec.input_ids.clone() if rec.input_ids is not None else None,
                                        attention_mask=rec.attention_mask.clone() if rec.attention_mask is not None else None,
                                        position_ids=rec.position_ids.clone() if rec.position_ids is not None else None,
                                        gt=rec.gt,
                                        reward=avg_r,
                                        est_reward=avg_r,
                                        meta={**(rec.meta or {}), "trained_round": self.global_steps},
                                        record_id=rec.record_id,
                                        original_text=rec.original_text,
                                        augmented_text=rec.augmented_text,
                                        teacher_response=rec.teacher_response,
                                        creation_time=rec.creation_time,
                                    )
                                    
                                    if self._validate_and_fix_record(archived_rec):
                                        self._trained_archive[rec.record_id] = archived_rec
                                        
                                        # Sync to index
                                        with self._inbox_lock:
                                            self._record_index[rec.record_id] = archived_rec
                                    
                                try:
                                    # Use SNAPSHOT values (captured BEFORE rollout) vs actual (AFTER rollout)
                                    driver_rewards = pre_rollout_snapshot["cached_rewards"]
                                    est_rewards = pre_rollout_snapshot["est_rewards"]
                                    origins = pre_rollout_snapshot["origins"]
                                    
                                    # Actual rewards from THIS rollout
                                    actual_rewards = [uid2rewards[base_uid_order[i]][0] if i < len(base_uid_order) and base_uid_order[i] in uid2rewards else None 
                                                    for i in range(len(sampled_records_for_this_step))]
                                    
                                    print(f"\n[REWARD-DEBUG] Step {self.global_steps} Reward Comparison Analysis")
                                    print(f"[REWARD-DEBUG] Comparing PRE-rollout state vs POST-rollout results")
                                    print(f"=" * 70)
                                    
                                    # ========== 1. Cached Rewards (from previous training) ==========
                                    valid_driver = [d for d in driver_rewards if d is not None and np.isfinite(d)]
                                    nan_driver_count = sum(1 for r in driver_rewards if r is None or (isinstance(r, float) and np.isnan(r)))
                                    
                                    print(f"\n1. CACHED REWARDS (what records had BEFORE this rollout):")
                                    print(f"   Source: Previous training round (or None if first scoring)")
                                    print(f"   Valid: {len(valid_driver)}/{len(driver_rewards)} (NaN/None: {nan_driver_count})")
                                    if valid_driver:
                                        print(f"   Mean: {np.mean(valid_driver):.3f}, Std: {np.std(valid_driver):.3f}")
                                        print(f"   Range: [{np.min(valid_driver):.3f}, {np.max(valid_driver):.3f}]")
                                    
                                    # ========== 2. Estimated Rewards (policy difficulty-adjusted) ==========
                                    valid_est = [e for e in est_rewards if e is not None and np.isfinite(e)]
                                    nan_est_count = sum(1 for r in est_rewards if r is None or (isinstance(r, float) and np.isnan(r)))
                                    
                                    print(f"\n2. ESTIMATED REWARDS (policy's difficulty prediction BEFORE rollout):")
                                    print(f"   Source: Policy-generated augmentations only")
                                    print(f"   Valid: {len(valid_est)}/{len(est_rewards)} (NaN/None: {nan_est_count})")
                                    if valid_est:
                                        print(f"   Mean: {np.mean(valid_est):.3f}, Std: {np.std(valid_est):.3f}")
                                        print(f"   Range: [{np.min(valid_est):.3f}, {np.max(valid_est):.3f}]")
                                    
                                    # ========== 3. Actual Rewards (from this rollout) ==========
                                    valid_actual = [a for a in actual_rewards if a is not None and np.isfinite(a)]
                                    
                                    print(f"\n3. ACTUAL REWARDS (scored during this rollout):")
                                    print(f"   Source: Current step's model + reward function")
                                    print(f"   Valid: {len(valid_actual)}/{len(actual_rewards)}")
                                    if valid_actual:
                                        print(f"   Mean: {np.mean(valid_actual):.3f}, Std: {np.std(valid_actual):.3f}")
                                        print(f"   Range: [{np.min(valid_actual):.3f}, {np.max(valid_actual):.3f}]")
                                    
                                    # ========== 4. Correlation Analysis ==========
                                    print(f"\n4. CORRELATION ANALYSIS:")
                                    
                                    # Driver vs Actual (for reinserted items)
                                    driver_actual_pairs = [(d, a) for d, a in zip(driver_rewards, actual_rewards) 
                                                        if d is not None and np.isfinite(d) and a is not None and np.isfinite(a)]
                                    if len(driver_actual_pairs) > 10:
                                        drivers_arr = np.array([p[0] for p in driver_actual_pairs])
                                        actuals_arr = np.array([p[1] for p in driver_actual_pairs])
                                        corr_driver_actual = np.corrcoef(drivers_arr, actuals_arr)[0, 1]
                                        mae_driver_actual = np.mean(np.abs(drivers_arr - actuals_arr))
                                        print(f"   Cached vs Actual:")
                                        print(f"     Correlation: {corr_driver_actual:.3f}")
                                        print(f"     MAE: {mae_driver_actual:.3f}")
                                        print(f"     Pairs: {len(driver_actual_pairs)}")
                                    else:
                                        print(f"   Cached vs Actual: Not enough pairs ({len(driver_actual_pairs)} < 10)")
                                    
                                    # Est vs Actual (for augmented queries with policy estimates)
                                    est_actual_pairs = [(e, a) for e, a in zip(est_rewards, actual_rewards) 
                                                    if e is not None and np.isfinite(e) and a is not None and np.isfinite(a)]
                                    if len(est_actual_pairs) > 10:
                                        est_arr = np.array([p[0] for p in est_actual_pairs])
                                        actuals_arr = np.array([p[1] for p in est_actual_pairs])
                                        corr_est_actual = np.corrcoef(est_arr, actuals_arr)[0, 1]
                                        mae_est_actual = np.mean(np.abs(est_arr - actuals_arr))
                                        print(f"   Estimated vs Actual:")
                                        print(f"     Correlation: {corr_est_actual:.3f}")
                                        print(f"     MAE: {mae_est_actual:.3f}")
                                        print(f"     Pairs: {len(est_actual_pairs)}")
                                        
                                        # Breakdown by origin
                                        est_by_origin = defaultdict(list)
                                        actual_by_origin = defaultdict(list)
                                        for i, (e, a) in enumerate(zip(est_rewards, actual_rewards)):
                                            if e is not None and np.isfinite(e) and a is not None and np.isfinite(a):
                                                origin = origins[i] if i < len(origins) else "unknown"
                                                est_by_origin[origin].append(e)
                                                actual_by_origin[origin].append(a)
                                        
                                        print(f"     By origin:")
                                        for origin in est_by_origin.keys():
                                            if len(est_by_origin[origin]) > 5:
                                                e_arr = np.array(est_by_origin[origin])
                                                a_arr = np.array(actual_by_origin[origin])
                                                corr = np.corrcoef(e_arr, a_arr)[0, 1]
                                                mae = np.mean(np.abs(e_arr - a_arr))
                                                print(f"       {origin}: corr={corr:.3f}, mae={mae:.3f}, n={len(e_arr)}")
                                    else:
                                        print(f"   Estimated vs Actual: Not enough pairs ({len(est_actual_pairs)} < 10)")
                                    
                                    # ========== 5. Delta Analysis (how much rewards changed) ==========
                                    print(f"\n5. REWARD DELTAS:")
                                    
                                    # For items that had previous rewards
                                    driver_deltas = [(a - d) for d, a in zip(driver_rewards, actual_rewards) 
                                                    if d is not None and np.isfinite(d) and a is not None and np.isfinite(a)]
                                    if driver_deltas:
                                        print(f"   Cached→Actual delta (reinserted items):")
                                        print(f"     Mean: {np.mean(driver_deltas):+.3f}, Std: {np.std(driver_deltas):.3f}")
                                        print(f"     Range: [{np.min(driver_deltas):+.3f}, {np.max(driver_deltas):+.3f}]")
                                        
                                        # Improvement/degradation counts
                                        improved = sum(1 for d in driver_deltas if d > 0.05)
                                        degraded = sum(1 for d in driver_deltas if d < -0.05)
                                        stable = len(driver_deltas) - improved - degraded
                                        print(f"     Improved: {improved}, Stable: {stable}, Degraded: {degraded}")
                                    
                                    # For augmented items with estimates
                                    est_deltas = [(a - e) for e, a in zip(est_rewards, actual_rewards) 
                                                if e is not None and np.isfinite(e) and a is not None and np.isfinite(a)]
                                    if est_deltas:
                                        print(f"   Estimated→Actual delta (augmented items):")
                                        print(f"     Mean: {np.mean(est_deltas):+.3f}, Std: {np.std(est_deltas):.3f}")
                                        print(f"     Range: [{np.min(est_deltas):+.3f}, {np.max(est_deltas):+.3f}]")
                                        
                                        # Over/under estimation
                                        overestimated = sum(1 for d in est_deltas if d < -0.05)
                                        underestimated = sum(1 for d in est_deltas if d > 0.05)
                                        accurate = len(est_deltas) - overestimated - underestimated
                                        print(f"     Overestimated: {overestimated}, Accurate: {accurate}, Underestimated: {underestimated}")
                                    
                                    print(f"=" * 70 + "\n")

                                except Exception as e:
                                    print(f"[REWARD-DEBUG] Error in reward comparison: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    
                                # Log metrics for tracking over time
                                try:
                                    if len(driver_actual_pairs) > 10:
                                        metrics["reward_analysis/cached_actual_correlation"] = corr_driver_actual
                                        metrics["reward_analysis/cached_actual_mae"] = mae_driver_actual
                                        metrics["reward_analysis/cached_actual_pairs"] = len(driver_actual_pairs)

                                    if len(est_actual_pairs) > 10:
                                        metrics["reward_analysis/est_actual_correlation"] = corr_est_actual
                                        metrics["reward_analysis/est_actual_mae"] = mae_est_actual
                                        metrics["reward_analysis/est_actual_pairs"] = len(est_actual_pairs)
                                    
                                    if driver_deltas:
                                        metrics["reward_analysis/cached_delta_mean"] = float(np.mean(driver_deltas))
                                        metrics["reward_analysis/cached_delta_std"] = float(np.std(driver_deltas))
                                        metrics["reward_analysis/items_improved"] = improved
                                        metrics["reward_analysis/items_degraded"] = degraded
                                    
                                    if est_deltas:
                                        metrics["reward_analysis/est_delta_mean"] = float(np.mean(est_deltas))
                                        metrics["reward_analysis/est_delta_std"] = float(np.std(est_deltas))
                                        metrics["reward_analysis/est_overestimated"] = overestimated
                                        metrics["reward_analysis/est_underestimated"] = underestimated
                                        
                                except Exception as e:
                                    logger.debug(f"Failed to log reward analysis metrics: {e}")
    
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

                        print("[FIT-DEBUG-6] Reward calculation completed. Start augmentation...")
                        # Step 4: policy-driven augmentation (THROTTLED)
                        if do_augment:
                            try:
                                self.query_pool.set_max_size(
                                    int(getattr(self.config.dynamic_data, "max_pool_size", 30000)))
                                remain = self.query_pool.capacity_remaining()

                                # Desired per-prompt augmentation
                                want_per_prompt = int(
                                    aug_cfg.get("num_per_prompt", 1))

                                # Number of ORIGINAL prompts in this step (before rollout repeats).
                                # We already computed these above as base_bsz; recompute defensively if needed.
                                try:
                                    num_prompts = len(gen_batch_unrepeated.batch["input_ids"])
                                    print("[FIT-DEBUG-7] Pre-augmentation Number of original prompts:", num_prompts)
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

                                # Strict capacity gate: only augment if we can fit ALL new queries.
                                required = want_per_prompt * num_prompts

                                print("[FIT-DEBUG-8] Augmentation capacity check:", {
                                    "want_per_prompt": want_per_prompt,
                                    "num_prompts": num_prompts,
                                    "required_capacity": required,
                                    "capacity_remaining": remain,
                                })
                                if want_per_prompt > 0 and num_prompts > 0 and remain >= required:
                                    # NEW: pass base reward overrides (actual per-prompt reward)
                                    aug_cfg_this = dict(aug_cfg or {})
                                    aug_cfg_this["base_reward_overrides_hash"] = base_reward_overrides_hash
                                    print(f"\n[AUG-FLOW-2] CALLING AUGMENTATION at step {self.global_steps}")
                                    print(f"  num_prompts: {num_prompts}, want_per_prompt: {want_per_prompt}")
                                    aug_proto = self.generate_augmented_queries(
                                        source_batch=gen_batch_unrepeated,
                                        num_per_prompt=want_per_prompt,
                                        aug_cfg=aug_cfg_this,
                                    )
                                    aug_size = len(aug_proto.batch["input_ids"])
                                    print(f"\n[AUG-FLOW-3] ENQUEUING TO TEACHER")
                                    print(f"  aug_proto size: {aug_size}")
                                    print(f"  Teacher queue size before: {self.teacher_annotator.queue.qsize()}")
                                    if self.teacher_annotator is not None and len(aug_proto.batch["input_ids"]) > 0:
                                        if not self.teacher_annotator.enqueue_aug(aug_proto):
                                            metrics["augmentation/queue_full_events"] = metrics.get(
                                                "augmentation/queue_full_events", 0) + 1
                                        print(f"  Teacher queue size after: {self.teacher_annotator.queue.qsize()}")
                                else:
                                    # Log why we skipped (helps debugging)
                                    metrics["augmentation/skipped_due_to_capacity"] = metrics.get(
                                        "augmentation/skipped_due_to_capacity", 0) + 1
                                    metrics["augmentation/required_capacity"] = required
                                    metrics["augmentation/capacity_remaining"] = remain
                                    metrics["augmentation/num_prompts"] = num_prompts
                                    metrics["augmentation/num_per_prompt"] = want_per_prompt

                            except Exception as e:
                                print(f"[dynamic] augmentation error: {e}")

                        print("[FIT-DEBUG-9] Augmentation step completed. Start group filtering...")
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
                            prompt_uid2traj_indices = defaultdict(list)  # NEW: Track trajectory indices
                            
                            for idx, (uid, metric_val) in enumerate(zip(
                                new_batch.non_tensor_batch["uid"],
                                new_batch.non_tensor_batch[metric_name]
                            )):
                                prompt_uid2metric_vals[uid].append(metric_val)
                                prompt_uid2traj_indices[uid].append(idx)  # NEW: Store index

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
                                print(
                                    f"[GROUP-FILTER-DIAG] Not enough prompts yet: {num_prompt_in_batch}/{prompt_bsz}. "
                                    f"Will generate more (gen_batch={num_gen_batches})...", flush=True
                                )
                                max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f"{num_gen_batches=}. Keep generating...", flush=True)
                                    continue
                                else:
                                    print(
                                        f"[GROUP-FILTER-DIAG] GIVING UP: {num_gen_batches=} >= {max_num_gen_batches=}. "
                                        f"Only collected {num_prompt_in_batch}/{prompt_bsz} prompts.", flush=True
                                    )
                                    raise ValueError(
                                        f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                        + " Generated too many. Please check if your data are too difficult."
                                        + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                    )
                            else:
                                traj_bsz = self.config.data.train_batch_size * \
                                    self.config.actor_rollout_ref.rollout.n
                                print(
                                    f"[GROUP-FILTER-DIAG] SUCCESS: Collected {num_prompt_in_batch} prompts. "
                                    f"Trimming batch to {traj_bsz} trajectories and proceeding to PPO update.", flush=True
                                )
                                batch = batch[:traj_bsz]

                        print("[GROUP-FILTER-DIAG] Group filtering completed. Start PPO update...")
                        
                        # Step 7: PPO updating
                        batch.batch["response_mask"] = compute_response_mask(batch)  # Fallback
                        
                        # ===== NEW: Save rollout records =====
                        save_freq = self.config.trainer.get("save_rollout_records_freq", 0)
                        if save_freq > 0 and (self.global_steps % save_freq == 0 or is_last_step):
                            self._save_rollout_records(batch, self.global_steps)
                            print("[FIT-DEBUG] Saved rollout records at step", self.global_steps)
                        # ===== END: Save rollout records =====

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
                            if dyn_en:
                                _ckpt_dir = os.path.join(
                                    self.config.trainer.default_local_dir,
                                    f"global_step_{self.global_steps}"
                                )
                                self._save_training_state(_ckpt_dir)

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
                    if dyn_en and self.query_pool:
                        pool_metrics = self.query_pool.get_metrics()
                        for k, v in pool_metrics.items():
                            metrics[f"pool/{k}"] = v
                        
                        # Additional prioritized sampling metrics
                        with self.query_pool._lock:
                            if self.query_pool._success_rates:
                                rates = list(self.query_pool._success_rates.values())
                                metrics["pool/success_rate_mean"] = float(np.mean(rates))
                                metrics["pool/success_rate_std"] = float(np.std(rates))
                                metrics["pool/success_rate_min"] = float(np.min(rates))
                                metrics["pool/success_rate_max"] = float(np.max(rates))
                                
                                # Distribution quartiles
                                sorted_rates = np.sort(rates)
                                n = len(sorted_rates)
                                metrics["pool/success_rate_q25"] = float(sorted_rates[n // 4])
                                metrics["pool/success_rate_q50"] = float(sorted_rates[n // 2])
                                metrics["pool/success_rate_q75"] = float(sorted_rates[3 * n // 4])

                    # Add augmentation metrics
                    for k, v in self._augmentation_metrics.items():
                        metrics[f"augmentation/{k}"] = v

                    # Add teacher annotator metrics if available
                    if self.teacher_annotator:
                        teacher_metrics = self.teacher_annotator.get_metrics()
                        for k, v in teacher_metrics.items():
                            metrics[f"teacher/{k}"] = v

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
                                # default to "original"
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
                    if dyn_en:
                        self._save_training_state(checkpoint_dir)
                metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
                logger_instance.log(data=metrics, step=self.global_steps)

        except Exception as e:
            print(f"Training loop error: {e}")
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
                print(
                    f"Error during teacher annotator shutdown(): {e}")
            try:
                teacher.join(timeout=10.0)
                if teacher.is_alive():
                    print(
                        "Teacher annotator thread did not terminate cleanly")
            except Exception as e:
                print(f"Error joining teacher annotator thread: {e}")
            self.teacher_annotator = None
            logger.info("Teacher annotator shutdown complete")

        # Final save of augmentation logs (guarded)
        aug_logger = getattr(self, "augmentation_logger", None)
        if aug_logger is not None:
            try:
                aug_logger.flush_all()
                logger.info("All augmentation logs saved")
            except Exception as e:
                print(f"Error flushing augmentation logs: {e}")