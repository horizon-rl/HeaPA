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
from contextlib import contextmanager
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
# Origin-based Reward/Advantage Analysis
# ==============================

@dataclass
class OriginAnalysisRecord:
    """Record for tracking rewards/advantages by origin."""
    step: int
    record_id: str
    uid: str
    origin: str  # "seed", "augmented", or "unknown"
    
    # Reward metrics
    reward: float
    reward_per_token: float
    
    # Advantage metrics  
    advantage: float
    advantage_abs: float
    
    # Value estimates (if using critic)
    return_value: Optional[float] = None
    value_estimate: Optional[float] = None
    value_error: Optional[float] = None
    
    # Sequence info
    sequence_length: int = 0
    prompt_length: int = 0
    response_length: int = 0
    
    # Task info
    ground_truth: Optional[str] = None
    is_correct: Optional[bool] = None
    
    # Augmentation info (for augmented queries)
    difficulty_factor: Optional[float] = None
    parent_id: Optional[str] = None
    teacher_difficulty: Optional[float] = None
    est_reward: Optional[float] = None
    
    # Timing
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "record_id": self.record_id,
            "uid": self.uid,
            "origin": self.origin,
            "reward": self.reward,
            "reward_per_token": self.reward_per_token,
            "advantage": self.advantage,
            "advantage_abs": self.advantage_abs,
            "return_value": self.return_value,
            "value_estimate": self.value_estimate,
            "value_error": self.value_error,
            "sequence_length": self.sequence_length,
            "prompt_length": self.prompt_length,
            "response_length": self.response_length,
            "ground_truth": self.ground_truth,
            "is_correct": self.is_correct,
            "difficulty_factor": self.difficulty_factor,
            "parent_id": self.parent_id,
            "teacher_difficulty": self.teacher_difficulty,
            "est_reward": self.est_reward,
            "timestamp": self.timestamp,
        }


class OriginAnalysisLogger:
    """
    Logs reward/advantage analysis split by origin (seed vs augmented).
    
    Saves detailed per-item records and summary statistics for comparing
    the difficulty and training contribution of seed vs augmented queries.
    """
    
    def __init__(self, log_dir: str, experiment_name: str, buffer_size: int = 500):
        self.log_dir = Path(log_dir) / "origin_analysis"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Separate files for detailed records by origin
        self.seed_file = self.log_dir / f"{experiment_name}_{self.timestamp}_seed_records.jsonl"
        self.augmented_file = self.log_dir / f"{experiment_name}_{self.timestamp}_augmented_records.jsonl"
        
        # Summary file (periodic snapshots)
        self.summary_file = self.log_dir / f"{experiment_name}_{self.timestamp}_origin_summary.jsonl"
        
        # Comparison file (side-by-side stats)
        self.comparison_file = self.log_dir / f"{experiment_name}_{self.timestamp}_comparison.jsonl"
        
        # Distribution files (for histogram visualization)
        self.distribution_dir = self.log_dir / "distributions"
        self.distribution_dir.mkdir(parents=True, exist_ok=True)
        
        # Buffers
        self.seed_buffer: List[Dict] = []
        self.augmented_buffer: List[Dict] = []
        self.buffer_size = buffer_size
        self._lock = threading.Lock()
        
        # Running statistics by origin
        self.stats = {
            "seed": self._init_origin_stats(),
            "augmented": self._init_origin_stats(),
        }
        
        # Per-step aggregates for time-series analysis
        self.step_aggregates: List[Dict] = []
        self.current_step_data = {"seed": [], "augmented": []}
        self.current_step = 0
        
        # Create metadata file
        metadata = {
            "experiment_name": experiment_name,
            "timestamp": self.timestamp,
            "files": {
                "seed_records": str(self.seed_file),
                "augmented_records": str(self.augmented_file),
                "summary": str(self.summary_file),
                "comparison": str(self.comparison_file),
            }
        }
        with open(self.log_dir / f"{experiment_name}_{self.timestamp}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"OriginAnalysisLogger initialized at {self.log_dir}")
    
    @staticmethod
    def _init_origin_stats() -> Dict[str, Any]:
        """Initialize statistics dictionary for an origin."""
        return {
            "count": 0,
            "reward_sum": 0.0,
            "reward_sq_sum": 0.0,
            "reward_min": float('inf'),
            "reward_max": float('-inf'),
            "advantage_sum": 0.0,
            "advantage_sq_sum": 0.0,
            "advantage_abs_sum": 0.0,
            "advantage_min": float('inf'),
            "advantage_max": float('-inf'),
            "return_sum": 0.0,
            "return_sq_sum": 0.0,
            "value_error_sum": 0.0,
            "value_error_sq_sum": 0.0,
            "correct_count": 0,
            "scored_count": 0,
            "total_tokens": 0,
            "total_response_tokens": 0,
            # Reward buckets for distribution
            "reward_buckets": defaultdict(int),  # bucket -> count
            "advantage_buckets": defaultdict(int),
        }
    
    def _bucket_value(self, value: float, bucket_size: float = 0.1) -> str:
        """Convert a value to a bucket string for histogram."""
        if not np.isfinite(value):
            return "nan"
        bucket_idx = int(np.floor(value / bucket_size))
        return f"{bucket_idx * bucket_size:.2f}"
    
    def log_record(self, record: OriginAnalysisRecord):
        """Log a single analysis record."""
        origin = record.origin if record.origin in ("seed", "augmented") else "seed"
        record_dict = record.to_dict()
        record_dict["timestamp"] = datetime.now().isoformat()
        
        with self._lock:
            # Update running stats
            stats = self.stats[origin]
            stats["count"] += 1
            
            # Reward stats
            reward = record.reward
            if reward is not None and np.isfinite(reward):
                stats["reward_sum"] += reward
                stats["reward_sq_sum"] += reward ** 2
                stats["reward_min"] = min(stats["reward_min"], reward)
                stats["reward_max"] = max(stats["reward_max"], reward)
                stats["reward_buckets"][self._bucket_value(reward)] += 1
            
            # Advantage stats
            advantage = record.advantage
            if advantage is not None and np.isfinite(advantage):
                stats["advantage_sum"] += advantage
                stats["advantage_sq_sum"] += advantage ** 2
                stats["advantage_abs_sum"] += abs(advantage)
                stats["advantage_min"] = min(stats["advantage_min"], advantage)
                stats["advantage_max"] = max(stats["advantage_max"], advantage)
                stats["advantage_buckets"][self._bucket_value(advantage)] += 1
            
            # Return/value stats
            if record.return_value is not None and np.isfinite(record.return_value):
                stats["return_sum"] += record.return_value
                stats["return_sq_sum"] += record.return_value ** 2
            
            if record.value_error is not None and np.isfinite(record.value_error):
                stats["value_error_sum"] += record.value_error
                stats["value_error_sq_sum"] += record.value_error ** 2
            
            # Correctness stats
            if record.is_correct is not None:
                stats["scored_count"] += 1
                if record.is_correct:
                    stats["correct_count"] += 1
            
            # Token stats
            stats["total_tokens"] += record.sequence_length
            stats["total_response_tokens"] += record.response_length
            
            # Per-step tracking
            self.current_step_data[origin].append(record_dict)
            
            # Add to buffer
            if origin == "seed":
                self.seed_buffer.append(record_dict)
                if len(self.seed_buffer) >= self.buffer_size:
                    self._flush_buffer("seed")
            else:
                self.augmented_buffer.append(record_dict)
                if len(self.augmented_buffer) >= self.buffer_size:
                    self._flush_buffer("augmented")
    
    def log_batch(self, batch: DataProto, step: int, use_critic: bool = False):
        """
        Extract and log records from a training batch.
        
        Args:
            batch: DataProto containing the training batch with rewards/advantages
            step: Current training step
            use_critic: Whether critic values are available
        """
        # Check if we have the required data
        if "token_level_rewards" not in batch.batch:
            logger.warning("Cannot log origin analysis: missing token_level_rewards")
            return
        
        # Update current step tracking
        if step != self.current_step:
            # Finalize previous step
            if self.current_step > 0:
                self._finalize_step(self.current_step)
            self.current_step = step
            self.current_step_data = {"seed": [], "augmented": []}
        
        n = len(batch.batch["token_level_rewards"])
        
        # Extract arrays
        rewards = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
        
        # Get advantages if available
        if "advantages" in batch.batch:
            advantages = batch.batch["advantages"]
            # Handle different advantage shapes
            if advantages.dim() > 1:
                advantages = advantages.sum(dim=-1)
            advantages = advantages.detach().cpu().numpy()
        else:
            advantages = np.zeros(n)
        
        # Get returns and values if using critic
        returns = None
        values = None
        if use_critic:
            if "returns" in batch.batch:
                returns = batch.batch["returns"]
                if returns.dim() > 1:
                    returns = returns.mean(dim=-1)
                returns = returns.detach().cpu().numpy()
            if "values" in batch.batch:
                values = batch.batch["values"]
                if values.dim() > 1:
                    values = values.mean(dim=-1)
                values = values.detach().cpu().numpy()
        
        # Get response mask for length calculation
        response_mask = batch.batch.get("response_mask")
        if response_mask is not None:
            response_lengths = response_mask.sum(dim=-1).detach().cpu().numpy()
        else:
            response_lengths = np.zeros(n)
        
        # Get sequence lengths
        attention_mask = batch.batch.get("attention_mask")
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=-1).detach().cpu().numpy()
        else:
            seq_lengths = np.zeros(n)
        
        # Get non-tensor metadata
        nt = batch.non_tensor_batch
        
        # Origin detection
        origins = list(nt.get("origin", ["seed"] * n))
        if len(origins) < n:
            origins.extend(["seed"] * (n - len(origins)))
        
        # Fallback: check is_augmented flag
        is_augmented = list(nt.get("is_augmented", [False] * n))
        for i in range(n):
            if i < len(is_augmented) and is_augmented[i]:
                origins[i] = "augmented"
        
        # Record IDs
        record_ids = list(nt.get("record_ids", [str(uuid.uuid4()) for _ in range(n)]))
        uids = list(nt.get("uid", record_ids))
        
        # Ground truth and correctness
        reward_model = list(nt.get("reward_model", [{}] * n))
        reward_extra = nt.get("reward_extra_info", {})
        is_correct_list = list(reward_extra.get("is_correct", [None] * n)) if isinstance(reward_extra, dict) else [None] * n
        
        # Augmentation metadata
        difficulty_factors = list(nt.get("difficulty_factors", [None] * n))
        parent_ids = list(nt.get("parent_record_id", [None] * n))
        teacher_diffs = list(nt.get("teacher/difficulty", [None] * n))
        est_rewards = list(nt.get("policy/est_reward", nt.get("driver_est_reward", [None] * n)))
        
        # Create records
        for i in range(n):
            origin = str(origins[i]) if i < len(origins) else "seed"
            if origin not in ("seed", "augmented"):
                origin = "augmented" if "augment" in origin.lower() else "seed"
            
            # Get ground truth
            gt = None
            if i < len(reward_model) and isinstance(reward_model[i], dict):
                gt = reward_model[i].get("ground_truth")
            
            # Get correctness
            is_correct = None
            if i < len(is_correct_list) and is_correct_list[i] is not None:
                is_correct = bool(is_correct_list[i])
            
            # Calculate value error if we have both
            value_error = None
            if returns is not None and values is not None:
                if np.isfinite(returns[i]) and np.isfinite(values[i]):
                    value_error = returns[i] - values[i]
            
            # Response tokens
            resp_len = int(response_lengths[i]) if i < len(response_lengths) else 0
            seq_len = int(seq_lengths[i]) if i < len(seq_lengths) else 0
            prompt_len = seq_len - resp_len
            
            # Reward per token
            reward_per_token = rewards[i] / max(resp_len, 1) if resp_len > 0 else rewards[i]
            
            record = OriginAnalysisRecord(
                step=step,
                record_id=str(record_ids[i]) if i < len(record_ids) else str(uuid.uuid4()),
                uid=str(uids[i]) if i < len(uids) else "",
                origin=origin,
                reward=float(rewards[i]),
                reward_per_token=float(reward_per_token),
                advantage=float(advantages[i]),
                advantage_abs=float(abs(advantages[i])),
                return_value=float(returns[i]) if returns is not None and np.isfinite(returns[i]) else None,
                value_estimate=float(values[i]) if values is not None and np.isfinite(values[i]) else None,
                value_error=value_error,
                sequence_length=seq_len,
                prompt_length=prompt_len,
                response_length=resp_len,
                ground_truth=str(gt) if gt is not None else None,
                is_correct=is_correct,
                difficulty_factor=float(difficulty_factors[i]) if i < len(difficulty_factors) and difficulty_factors[i] is not None else None,
                parent_id=str(parent_ids[i]) if i < len(parent_ids) and parent_ids[i] is not None else None,
                teacher_difficulty=float(teacher_diffs[i]) if i < len(teacher_diffs) and teacher_diffs[i] is not None else None,
                est_reward=float(est_rewards[i]) if i < len(est_rewards) and est_rewards[i] is not None and np.isfinite(est_rewards[i]) else None,
            )
            
            self.log_record(record)
    
    def _finalize_step(self, step: int):
        """Finalize statistics for a step and save comparison."""
        with self._lock:
            seed_data = self.current_step_data["seed"]
            aug_data = self.current_step_data["augmented"]
            
            def compute_step_stats(data: List[Dict]) -> Dict[str, Any]:
                if not data:
                    return {"count": 0}
                
                rewards = [d["reward"] for d in data if d.get("reward") is not None and np.isfinite(d["reward"])]
                advantages = [d["advantage"] for d in data if d.get("advantage") is not None and np.isfinite(d["advantage"])]
                
                stats = {
                    "count": len(data),
                    "reward_count": len(rewards),
                    "advantage_count": len(advantages),
                }
                
                if rewards:
                    stats["reward_mean"] = float(np.mean(rewards))
                    stats["reward_std"] = float(np.std(rewards))
                    stats["reward_min"] = float(np.min(rewards))
                    stats["reward_max"] = float(np.max(rewards))
                    stats["reward_median"] = float(np.median(rewards))
                    # Percentiles
                    stats["reward_p25"] = float(np.percentile(rewards, 25))
                    stats["reward_p75"] = float(np.percentile(rewards, 75))
                
                if advantages:
                    stats["advantage_mean"] = float(np.mean(advantages))
                    stats["advantage_std"] = float(np.std(advantages))
                    stats["advantage_abs_mean"] = float(np.mean(np.abs(advantages)))
                    stats["advantage_min"] = float(np.min(advantages))
                    stats["advantage_max"] = float(np.max(advantages))
                
                # Correctness
                correct = [d["is_correct"] for d in data if d.get("is_correct") is not None]
                if correct:
                    stats["accuracy"] = float(sum(correct) / len(correct))
                    stats["scored_count"] = len(correct)
                
                return stats
            
            seed_stats = compute_step_stats(seed_data)
            aug_stats = compute_step_stats(aug_data)
            
            comparison = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "seed": seed_stats,
                "augmented": aug_stats,
            }
            
            # Compute deltas
            if seed_stats.get("reward_mean") is not None and aug_stats.get("reward_mean") is not None:
                comparison["delta_reward_mean"] = aug_stats["reward_mean"] - seed_stats["reward_mean"]
            if seed_stats.get("advantage_mean") is not None and aug_stats.get("advantage_mean") is not None:
                comparison["delta_advantage_mean"] = aug_stats["advantage_mean"] - seed_stats["advantage_mean"]
            if seed_stats.get("accuracy") is not None and aug_stats.get("accuracy") is not None:
                comparison["delta_accuracy"] = aug_stats["accuracy"] - seed_stats["accuracy"]
            
            # Save comparison
            try:
                with open(self.comparison_file, 'a') as f:
                    f.write(json.dumps(comparison) + '\n')
            except Exception as e:
                logger.warning(f"Failed to save comparison: {e}")
            
            self.step_aggregates.append(comparison)
    
    def _flush_buffer(self, origin: str):
        """Flush buffer to file."""
        buffer = self.seed_buffer if origin == "seed" else self.augmented_buffer
        file = self.seed_file if origin == "seed" else self.augmented_file
        
        if not buffer:
            return
        
        try:
            with open(file, 'a') as f:
                for record in buffer:
                    f.write(json.dumps(record, default=str) + '\n')
        except Exception as e:
            logger.warning(f"Failed to flush {origin} buffer: {e}")
        
        buffer.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current summary statistics."""
        with self._lock:
            def compute_summary(stats: Dict) -> Dict:
                n = stats["count"]
                if n == 0:
                    return {"count": 0}
                
                summary = {"count": n}
                
                # Reward stats
                if n > 0:
                    summary["reward_mean"] = stats["reward_sum"] / n
                    variance = (stats["reward_sq_sum"] / n) - (summary["reward_mean"] ** 2)
                    summary["reward_std"] = np.sqrt(max(0, variance))
                    if stats["reward_min"] != float('inf'):
                        summary["reward_min"] = stats["reward_min"]
                        summary["reward_max"] = stats["reward_max"]
                
                # Advantage stats
                if n > 0:
                    summary["advantage_mean"] = stats["advantage_sum"] / n
                    variance = (stats["advantage_sq_sum"] / n) - (summary["advantage_mean"] ** 2)
                    summary["advantage_std"] = np.sqrt(max(0, variance))
                    summary["advantage_abs_mean"] = stats["advantage_abs_sum"] / n
                    if stats["advantage_min"] != float('inf'):
                        summary["advantage_min"] = stats["advantage_min"]
                        summary["advantage_max"] = stats["advantage_max"]
                
                # Accuracy
                if stats["scored_count"] > 0:
                    summary["accuracy"] = stats["correct_count"] / stats["scored_count"]
                    summary["scored_count"] = stats["scored_count"]
                
                # Token stats
                summary["total_tokens"] = stats["total_tokens"]
                summary["avg_response_length"] = stats["total_response_tokens"] / n if n > 0 else 0
                
                return summary
            
            return {
                "seed": compute_summary(self.stats["seed"]),
                "augmented": compute_summary(self.stats["augmented"]),
                "timestamp": datetime.now().isoformat(),
            }
    
    def save_summary(self, step: int):
        """Save periodic summary to file."""
        summary = self.get_summary()
        summary["step"] = step
        
        try:
            with open(self.summary_file, 'a') as f:
                f.write(json.dumps(summary) + '\n')
        except Exception as e:
            logger.warning(f"Failed to save summary: {e}")
    
    def save_distributions(self, step: int):
        """Save reward/advantage distributions for visualization."""
        with self._lock:
            for origin in ["seed", "augmented"]:
                stats = self.stats[origin]
                
                # Save reward distribution
                reward_dist = dict(stats["reward_buckets"])
                if reward_dist:
                    dist_file = self.distribution_dir / f"{origin}_reward_dist_step{step}.json"
                    with open(dist_file, 'w') as f:
                        json.dump({
                            "step": step,
                            "origin": origin,
                            "metric": "reward",
                            "distribution": reward_dist,
                            "count": stats["count"],
                        }, f, indent=2)
                
                # Save advantage distribution
                adv_dist = dict(stats["advantage_buckets"])
                if adv_dist:
                    dist_file = self.distribution_dir / f"{origin}_advantage_dist_step{step}.json"
                    with open(dist_file, 'w') as f:
                        json.dump({
                            "step": step,
                            "origin": origin,
                            "metric": "advantage",
                            "distribution": adv_dist,
                            "count": stats["count"],
                        }, f, indent=2)
    
    def flush_all(self):
        """Flush all buffers and finalize current step."""
        with self._lock:
            if self.current_step > 0:
                self._finalize_step(self.current_step)
            self._flush_buffer("seed")
            self._flush_buffer("augmented")


# ==============================
# Efficiency/Timing Analysis with FLOP Estimation
# ==============================

@dataclass 
class TimingRecord:
    """Record for a single timing measurement."""
    step: int
    stage: str
    wall_time_seconds: float
    batch_size: int
    sequence_length: int
    new_tokens: int  # For generation stages
    estimated_tflops: float
    gpu_memory_mb: float
    throughput_tokens_per_sec: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "stage": self.stage,
            "wall_time_seconds": self.wall_time_seconds,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "new_tokens": self.new_tokens,
            "estimated_tflops": self.estimated_tflops,
            "gpu_memory_mb": self.gpu_memory_mb,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "timestamp": self.timestamp,
        }


class EfficiencyAnalyzer:
    """
    Tracks wall clock time and estimates FLOPs for training stages.
    
    Provides detailed timing breakdowns and FLOP estimates to understand
    the computational cost of different training stages, especially
    comparing augmentation overhead vs training benefit.
    """
    
    # Stage names for consistent tracking
    STAGE_ROLLOUT_GEN = "rollout_generation"
    STAGE_AUGMENT_GEN = "augmentation_generation"
    STAGE_REWARD_COMPUTE = "reward_computation"
    STAGE_ACTOR_UPDATE = "actor_update"
    STAGE_CRITIC_UPDATE = "critic_update"
    STAGE_REF_LOGPROB = "ref_log_prob"
    STAGE_OLD_LOGPROB = "old_log_prob"
    STAGE_ADVANTAGE = "advantage_computation"
    STAGE_TEACHER_ANNOTATE = "teacher_annotation"
    STAGE_POOL_SAMPLE = "pool_sampling"
    STAGE_TOTAL_STEP = "total_step"
    
    def __init__(
        self, 
        log_dir: str, 
        experiment_name: str,
        model_config: Any,
        buffer_size: int = 200
    ):
        self.log_dir = Path(log_dir) / "efficiency_analysis"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Output files
        self.timing_file = self.log_dir / f"{experiment_name}_{self.timestamp}_timing.jsonl"
        self.summary_file = self.log_dir / f"{experiment_name}_{self.timestamp}_efficiency_summary.jsonl"
        self.comparison_file = self.log_dir / f"{experiment_name}_{self.timestamp}_stage_comparison.jsonl"
        
        # Model parameters for FLOP estimation
        self.model_config = model_config
        self._extract_model_params()
        
        # Accumulated timing by stage
        self.stage_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_time": 0.0,
            "count": 0,
            "total_tflops": 0.0,
            "total_tokens": 0,
            "samples": [],  # Keep last N samples for analysis
            "max_samples": 100,
        })
        
        # Per-step timing for comparison
        self.step_timings: Dict[int, Dict[str, float]] = {}
        self.current_step = 0
        self.current_step_timings: Dict[str, float] = {}
        self.current_step_tflops: Dict[str, float] = {}
        
        self._lock = threading.Lock()
        self.buffer: List[Dict] = []
        self.buffer_size = buffer_size
        
        # Create metadata
        metadata = {
            "experiment_name": experiment_name,
            "timestamp": self.timestamp,
            "model_params": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "vocab_size": self.vocab_size,
                "intermediate_size": self.intermediate_size,
            },
            "files": {
                "timing": str(self.timing_file),
                "summary": str(self.summary_file),
                "comparison": str(self.comparison_file),
            }
        }
        with open(self.log_dir / f"{experiment_name}_{self.timestamp}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"EfficiencyAnalyzer initialized at {self.log_dir}")
    
    def _extract_model_params(self):
        """Extract model parameters for FLOP estimation."""
        # Try to extract from config
        try:
            model_cfg = getattr(self.model_config, "model", self.model_config)
            
            # Try different config structures
            self.hidden_size = getattr(model_cfg, "hidden_size", 
                                       getattr(model_cfg, "d_model", 4096))
            self.num_layers = getattr(model_cfg, "num_hidden_layers",
                                     getattr(model_cfg, "num_layers", 32))
            self.num_heads = getattr(model_cfg, "num_attention_heads",
                                    getattr(model_cfg, "n_head", 32))
            self.vocab_size = getattr(model_cfg, "vocab_size", 32000)
            self.intermediate_size = getattr(model_cfg, "intermediate_size",
                                            self.hidden_size * 4)
            
        except Exception as e:
            logger.warning(f"Could not extract model params, using defaults: {e}")
            # Reasonable defaults for ~7B model
            self.hidden_size = 4096
            self.num_layers = 32
            self.num_heads = 32
            self.vocab_size = 32000
            self.intermediate_size = 11008
    
    def estimate_forward_tflops(
        self, 
        batch_size: int, 
        seq_length: int,
        is_generation: bool = False,
        new_tokens: int = 0
    ) -> float:
        """
        Estimate TFLOPs for a forward pass.
        
        For generation, we estimate the cost of generating new_tokens autoregressively.
        For training forward pass, we estimate the full sequence forward cost.
        
        Based on: https://arxiv.org/abs/2001.08361 (Scaling Laws paper)
        FLOPs ≈ 2 * P * T for forward, where P = params, T = tokens
        
        More detailed: For transformer layer
        - Attention: 4 * batch * seq * hidden^2 + 4 * batch * seq^2 * hidden
        - FFN: 8 * batch * seq * hidden * intermediate
        - Per layer total: ~12 * batch * seq * hidden^2 (simplified)
        - Full model: num_layers * layer_flops + embedding_flops
        """
        H = self.hidden_size
        L = self.num_layers
        V = self.vocab_size
        I = self.intermediate_size
        
        if is_generation and new_tokens > 0:
            # Generation: for each new token, attend to all previous
            # Average context length during generation
            avg_context = seq_length + new_tokens / 2
            
            # Per-token generation cost (simplified)
            # Attention: 4 * H^2 + 4 * avg_context * H
            # FFN: 2 * 4 * H * I
            per_token_flops = L * (
                4 * H * H +  # Q, K, V, O projections
                4 * avg_context * H +  # Attention
                8 * H * I  # FFN
            )
            
            # Output projection
            per_token_flops += H * V
            
            total_flops = batch_size * new_tokens * per_token_flops
            
        else:
            # Full forward pass
            # Per layer: attention + FFN
            per_token_flops = L * (
                4 * H * H +  # Q, K, V, O projections
                4 * seq_length * H +  # Attention (quadratic in seq)
                8 * H * I  # FFN
            )
            
            # Embedding + output
            per_token_flops += H * V * 2  # input embed + output project
            
            total_flops = batch_size * seq_length * per_token_flops
        
        # Convert to TFLOPs
        return total_flops / 1e12
    
    def estimate_backward_tflops(
        self,
        batch_size: int,
        seq_length: int
    ) -> float:
        """
        Estimate TFLOPs for backward pass.
        Backward is roughly 2x forward for gradient computation.
        """
        forward_tflops = self.estimate_forward_tflops(batch_size, seq_length)
        return forward_tflops * 2
    
    def estimate_stage_tflops(
        self,
        stage: str,
        batch_size: int,
        seq_length: int,
        new_tokens: int = 0
    ) -> float:
        """Estimate TFLOPs for a given stage."""
        
        if stage == self.STAGE_ROLLOUT_GEN:
            # Generation: forward only, but autoregressive
            return self.estimate_forward_tflops(
                batch_size, seq_length, is_generation=True, new_tokens=new_tokens
            )
        
        elif stage == self.STAGE_AUGMENT_GEN:
            # Similar to rollout generation
            return self.estimate_forward_tflops(
                batch_size, seq_length, is_generation=True, new_tokens=new_tokens
            )
        
        elif stage == self.STAGE_ACTOR_UPDATE:
            # Forward + backward
            fwd = self.estimate_forward_tflops(batch_size, seq_length)
            bwd = self.estimate_backward_tflops(batch_size, seq_length)
            return fwd + bwd
        
        elif stage == self.STAGE_CRITIC_UPDATE:
            # Typically smaller model, estimate as 1/4 of actor
            fwd = self.estimate_forward_tflops(batch_size, seq_length) * 0.25
            bwd = self.estimate_backward_tflops(batch_size, seq_length) * 0.25
            return fwd + bwd
        
        elif stage in (self.STAGE_OLD_LOGPROB, self.STAGE_REF_LOGPROB):
            # Forward only
            return self.estimate_forward_tflops(batch_size, seq_length)
        
        elif stage == self.STAGE_REWARD_COMPUTE:
            # Much smaller, typically rule-based or small model
            return self.estimate_forward_tflops(batch_size, seq_length) * 0.1
        
        elif stage == self.STAGE_ADVANTAGE:
            # CPU-bound, negligible GPU FLOPs
            return 0.0
        
        else:
            # Default: estimate as forward pass
            return self.estimate_forward_tflops(batch_size, seq_length)
    
    def get_gpu_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 2)
        except Exception:
            pass
        return 0.0
    
    @contextmanager
    def measure_stage(
        self,
        stage: str,
        step: int,
        batch_size: int,
        seq_length: int,
        new_tokens: int = 0
    ):
        """
        Context manager to measure a training stage.
        
        Usage:
            with efficiency_analyzer.measure_stage("rollout_generation", step, 32, 2048, 512):
                output = model.generate(...)
        """
        # Update step tracking
        if step != self.current_step:
            self._finalize_step()
            self.current_step = step
            self.current_step_timings = {}
            self.current_step_tflops = {}
        
        # Record start
        start_time = time.perf_counter()
        start_memory = self.get_gpu_memory_mb()
        
        # Synchronize CUDA for accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        try:
            yield
        finally:
            # Synchronize again
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Record end
            end_time = time.perf_counter()
            end_memory = self.get_gpu_memory_mb()
            
            wall_time = end_time - start_time
            memory_used = max(end_memory, start_memory)
            
            # Estimate TFLOPs
            estimated_tflops = self.estimate_stage_tflops(
                stage, batch_size, seq_length, new_tokens
            )
            
            # Calculate throughput
            total_tokens = batch_size * (seq_length + new_tokens)
            throughput = total_tokens / wall_time if wall_time > 0 else 0
            
            # Create record
            record = TimingRecord(
                step=step,
                stage=stage,
                wall_time_seconds=wall_time,
                batch_size=batch_size,
                sequence_length=seq_length,
                new_tokens=new_tokens,
                estimated_tflops=estimated_tflops,
                gpu_memory_mb=memory_used,
                throughput_tokens_per_sec=throughput,
                timestamp=datetime.now().isoformat(),
            )
            
            self.log_timing(record)
    
    def log_timing(self, record: TimingRecord):
        """Log a timing record."""
        with self._lock:
            stage = record.stage
            
            # Update accumulated stats
            stats = self.stage_stats[stage]
            stats["total_time"] += record.wall_time_seconds
            stats["count"] += 1
            stats["total_tflops"] += record.estimated_tflops
            stats["total_tokens"] += record.batch_size * (record.sequence_length + record.new_tokens)
            
            # Keep recent samples
            stats["samples"].append(record.to_dict())
            if len(stats["samples"]) > stats["max_samples"]:
                stats["samples"].pop(0)
            
            # Update current step tracking
            self.current_step_timings[stage] = self.current_step_timings.get(stage, 0) + record.wall_time_seconds
            self.current_step_tflops[stage] = self.current_step_tflops.get(stage, 0) + record.estimated_tflops
            
            # Add to buffer
            self.buffer.append(record.to_dict())
            if len(self.buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _finalize_step(self):
        """Finalize timing for the current step."""
        if self.current_step <= 0:
            return
        
        with self._lock:
            # Calculate total step time
            total_time = sum(self.current_step_timings.values())
            total_tflops = sum(self.current_step_tflops.values())
            
            # Calculate percentages
            step_record = {
                "step": self.current_step,
                "timestamp": datetime.now().isoformat(),
                "total_time": total_time,
                "total_tflops": total_tflops,
                "stages": {},
            }
            
            for stage, time_val in self.current_step_timings.items():
                step_record["stages"][stage] = {
                    "time": time_val,
                    "time_pct": (time_val / total_time * 100) if total_time > 0 else 0,
                    "tflops": self.current_step_tflops.get(stage, 0),
                    "tflops_pct": (self.current_step_tflops.get(stage, 0) / total_tflops * 100) if total_tflops > 0 else 0,
                }
            
            # Calculate augmentation overhead
            aug_time = self.current_step_timings.get(self.STAGE_AUGMENT_GEN, 0)
            train_time = (
                self.current_step_timings.get(self.STAGE_ACTOR_UPDATE, 0) +
                self.current_step_timings.get(self.STAGE_CRITIC_UPDATE, 0)
            )
            
            step_record["augmentation_overhead_pct"] = (aug_time / total_time * 100) if total_time > 0 else 0
            step_record["training_time_pct"] = (train_time / total_time * 100) if total_time > 0 else 0
            step_record["generation_time_pct"] = (
                (self.current_step_timings.get(self.STAGE_ROLLOUT_GEN, 0) + aug_time) / total_time * 100
            ) if total_time > 0 else 0
            
            # Save comparison
            try:
                with open(self.comparison_file, 'a') as f:
                    f.write(json.dumps(step_record) + '\n')
            except Exception as e:
                logger.warning(f"Failed to save step comparison: {e}")
            
            self.step_timings[self.current_step] = step_record
    
    def _flush_buffer(self):
        """Flush timing buffer to file."""
        if not self.buffer:
            return
        
        try:
            with open(self.timing_file, 'a') as f:
                for record in self.buffer:
                    f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.warning(f"Failed to flush timing buffer: {e}")
        
        self.buffer.clear()
    
    def get_efficiency_summary(self) -> Dict[str, Any]:
        """Get efficiency summary across all stages."""
        with self._lock:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "stages": {},
                "totals": {
                    "total_time": 0.0,
                    "total_tflops": 0.0,
                    "total_samples": 0,
                },
            }
            
            for stage, stats in self.stage_stats.items():
                if stats["count"] == 0:
                    continue
                
                stage_summary = {
                    "count": stats["count"],
                    "total_time": stats["total_time"],
                    "avg_time": stats["total_time"] / stats["count"],
                    "total_tflops": stats["total_tflops"],
                    "avg_tflops": stats["total_tflops"] / stats["count"],
                    "total_tokens": stats["total_tokens"],
                    "avg_throughput": stats["total_tokens"] / stats["total_time"] if stats["total_time"] > 0 else 0,
                }
                
                summary["stages"][stage] = stage_summary
                summary["totals"]["total_time"] += stats["total_time"]
                summary["totals"]["total_tflops"] += stats["total_tflops"]
                summary["totals"]["total_samples"] += stats["count"]
            
            # Calculate percentages
            total_time = summary["totals"]["total_time"]
            total_tflops = summary["totals"]["total_tflops"]
            
            for stage in summary["stages"]:
                summary["stages"][stage]["time_pct"] = (
                    summary["stages"][stage]["total_time"] / total_time * 100
                ) if total_time > 0 else 0
                summary["stages"][stage]["tflops_pct"] = (
                    summary["stages"][stage]["total_tflops"] / total_tflops * 100
                ) if total_tflops > 0 else 0
            
            # Key efficiency metrics
            aug_time = self.stage_stats.get(self.STAGE_AUGMENT_GEN, {}).get("total_time", 0)
            rollout_time = self.stage_stats.get(self.STAGE_ROLLOUT_GEN, {}).get("total_time", 0)
            train_time = (
                self.stage_stats.get(self.STAGE_ACTOR_UPDATE, {}).get("total_time", 0) +
                self.stage_stats.get(self.STAGE_CRITIC_UPDATE, {}).get("total_time", 0)
            )
            
            summary["efficiency_metrics"] = {
                "augmentation_overhead_pct": (aug_time / total_time * 100) if total_time > 0 else 0,
                "generation_to_training_ratio": (rollout_time + aug_time) / train_time if train_time > 0 else 0,
                "augmentation_to_rollout_ratio": aug_time / rollout_time if rollout_time > 0 else 0,
            }
            
            return summary
    
    def save_summary(self, step: int = None):
        """Save efficiency summary to file."""
        # Finalize current step
        self._finalize_step()
        
        summary = self.get_efficiency_summary()
        if step is not None:
            summary["step"] = step
        
        try:
            with open(self.summary_file, 'a') as f:
                f.write(json.dumps(summary) + '\n')
        except Exception as e:
            logger.warning(f"Failed to save efficiency summary: {e}")
    
    def get_metrics_for_logging(self) -> Dict[str, float]:
        """Get metrics suitable for logging to wandb/tensorboard."""
        summary = self.get_efficiency_summary()
        
        metrics = {}
        
        # Stage-specific metrics
        for stage, stats in summary.get("stages", {}).items():
            stage_key = stage.replace("_", "/")
            metrics[f"efficiency/{stage_key}/avg_time"] = stats.get("avg_time", 0)
            metrics[f"efficiency/{stage_key}/time_pct"] = stats.get("time_pct", 0)
            metrics[f"efficiency/{stage_key}/avg_tflops"] = stats.get("avg_tflops", 0)
            metrics[f"efficiency/{stage_key}/throughput"] = stats.get("avg_throughput", 0)
        
        # Overall metrics
        eff = summary.get("efficiency_metrics", {})
        metrics["efficiency/augmentation_overhead_pct"] = eff.get("augmentation_overhead_pct", 0)
        metrics["efficiency/gen_to_train_ratio"] = eff.get("generation_to_training_ratio", 0)
        metrics["efficiency/aug_to_rollout_ratio"] = eff.get("augmentation_to_rollout_ratio", 0)
        
        return metrics
    
    def flush_all(self):
        """Flush all buffers and finalize."""
        self._finalize_step()
        with self._lock:
            self._flush_buffer()


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

    low_heap  : key = -reward (acts like max-heap on reward).
    high_heap : true min-heap **and** mirror max-heap with lazy deletion.

    Full-insert rule:
      - If pool full: insert the item according to its reward and then evict
        according to the normal policy. There is **no** special-case rejection
        of high-reward ("easy") items.
    """

    def __init__(
        self,
        max_size: int = 30000,
        low_fraction: float = 0.5,
        rng: Optional[np.random.Generator] = None,
        cleanup_frequency: int = 1000,  # Clean up stale entries every N operations
        mixed_easy_medium: bool = False,
        trainer_ref: Optional["RayPPOTrainer"] = None,
    ):
        self._lock = threading.RLock()  # Use RLock to prevent deadlocks
        self._max_size = max(1, int(max_size))  # Ensure at least 1
        self._low_fraction = float(np.clip(low_fraction, 0.05, 0.95))
        self._cleanup_frequency = cleanup_frequency
        self._operations_count = 0
        self._mixed_easy_medium = bool(mixed_easy_medium)
        self.trainer_ref = trainer_ref

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

    def _remove_high_heap_min_unlocked(self) -> bool:
        """Remove item with smallest reward from high heap (worst high-reward item)."""
        self._clean_high_min_top_unlocked()
        while self._high_heap_min:
            r, seq, evicted_rec = heapq.heappop(self._high_heap_min)  # ← Changed: capture record
            if seq in self._active_high:
                self._active_high.remove(seq)
                self._total_evicted += 1
                
                self._cleanup_record_index(evicted_rec)
                
                return True
        return False

    def _remove_low_heap_min_unlocked(self) -> bool:
        """Remove item with smallest reward from low heap (worst low-reward item)."""
        if not self._low_heap:
            return False
        
        # Find item with worst reward (largest -reward value)
        worst_i = max(range(len(self._low_heap)), key=lambda i: self._low_heap[i][0])
        _, _, evicted_rec = self._low_heap[worst_i]  # Capture record before deletion
        
        # Delete-index trick: swap with last, then pop
        self._low_heap[worst_i] = self._low_heap[-1]
        self._low_heap.pop()
        heapq.heapify(self._low_heap)
        self._total_evicted += 1
        
        # Clean up record index to prevent memory leak
        self._cleanup_record_index(evicted_rec)
        
        return True

    def add_many(self, items: List[QueryRecord]):
        """Add multiple records with comprehensive tracking."""
        if not items:
            return
        
        print(f"\n{'='*70}")
        print(f"[POOL-ADD-MANY] START - Attempting to add {len(items)} items")
        print(f"{'='*70}")
        
        # Pre-addition state
        with self._lock:
            initial_size = self.size()
            initial_capacity = self.capacity_remaining()
            initial_low = len(self._low_heap)
            initial_high = self._high_size_unlocked()
            initial_cold = len(self._cold_queue)
        
        print(f"[POOL-ADD-MANY] Initial State:")
        print(f"  Total size: {initial_size}")
        print(f"  Capacity remaining: {initial_capacity}")
        print(f"  Distribution: low={initial_low}, high={initial_high}, cold={initial_cold}")
        
        # Track all outcomes
        added_count = 0
        rejected_validation = 0
        rejected_full = 0
        rejection_details = {
            "not_ndarray": 0,
            "wrong_ndim": 0,
            "wrong_dtype": 0,
            "empty_array": 0,
            "validation_exception": 0,
        }
        
        # Sample rejected items for detailed logging
        rejected_samples = []
        
        with self._lock:
            for idx, it in enumerate(items):
                # === VALIDATION GATE ===
                try:
                    data = it.raw_prompt_data
                    
                    # Check 1: Must be ndarray
                    if not isinstance(data, np.ndarray):
                        rejected_validation += 1
                        rejection_details["not_ndarray"] += 1
                        if len(rejected_samples) < 5:
                            rejected_samples.append({
                                "idx": idx,
                                "reason": "not_ndarray",
                                "type": type(data).__name__,
                                "record_id": it.record_id,
                                "origin": (it.meta or {}).get("origin", "unknown"),
                            })
                        continue
                    
                    # Check 2: Must be 1-D
                    if data.ndim != 1:
                        rejected_validation += 1
                        rejection_details["wrong_ndim"] += 1
                        if len(rejected_samples) < 5:
                            rejected_samples.append({
                                "idx": idx,
                                "reason": "wrong_ndim",
                                "ndim": data.ndim,
                                "shape": data.shape,
                                "record_id": it.record_id,
                                "origin": (it.meta or {}).get("origin", "unknown"),
                            })
                        continue
                    
                    # Check 3: Must be integer dtype
                    if data.dtype == object or not np.issubdtype(data.dtype, np.integer):
                        rejected_validation += 1
                        rejection_details["wrong_dtype"] += 1
                        if len(rejected_samples) < 5:
                            rejected_samples.append({
                                "idx": idx,
                                "reason": "wrong_dtype",
                                "dtype": str(data.dtype),
                                "record_id": it.record_id,
                                "origin": (it.meta or {}).get("origin", "unknown"),
                            })
                        continue
                    
                    # Check 4: Must not be empty
                    if data.size == 0:
                        rejected_validation += 1
                        rejection_details["empty_array"] += 1
                        if len(rejected_samples) < 5:
                            rejected_samples.append({
                                "idx": idx,
                                "reason": "empty_array",
                                "record_id": it.record_id,
                                "origin": (it.meta or {}).get("origin", "unknown"),
                            })
                        continue
                        
                except Exception as e:
                    rejected_validation += 1
                    rejection_details["validation_exception"] += 1
                    if len(rejected_samples) < 5:
                        rejected_samples.append({
                            "idx": idx,
                            "reason": "validation_exception",
                            "error": str(e),
                            "record_id": it.record_id if hasattr(it, 'record_id') else "unknown",
                        })
                    continue
                
                # === CAPACITY LOGIC ===
                total = len(self._low_heap) + self._high_size_unlocked() + len(self._cold_queue)
                
                if total < self._max_size:
                    # Pool has space
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
                    # Pool full - try to insert
                    if it.reward is None:
                        # Cannot order; evict and add
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
                        # Try insertion when full
                        if self._insert_when_full_unlocked(it):
                            added_count += 1
                        else:
                            rejected_full += 1
            
            # Update global counter
            self._total_added += added_count
            self._operations_count += len(items)
            
            # Periodic cleanup
            if self._operations_count >= self._cleanup_frequency:
                self._deep_clean_heaps_unlocked()
                self._operations_count = 0
            
            # Enforce capacity
            self._evict_to_capacity_unlocked()
            
            # Post-addition state
            final_size = self.size()
            final_capacity = self.capacity_remaining()
            final_low = len(self._low_heap)
            final_high = self._high_size_unlocked()
            final_cold = len(self._cold_queue)
        
        # === SUMMARY LOGGING ===
        print(f"\n[POOL-ADD-MANY] Results:")
        print(f"  ✓ Added: {added_count}")
        print(f"  ✗ Rejected (validation): {rejected_validation}")
        print(f"  ✗ Rejected (pool full): {rejected_full}")
        print(f"  Total processed: {len(items)}")
        
        if rejected_validation > 0:
            print(f"\n[POOL-ADD-MANY] Rejection Breakdown:")
            for reason, count in rejection_details.items():
                if count > 0:
                    print(f"    {reason}: {count}")
            
            if rejected_samples:
                print(f"\n[POOL-ADD-MANY] Sample Rejected Items (first {len(rejected_samples)}):")
                for sample in rejected_samples:
                    print(f"    Item {sample['idx']}:")
                    for k, v in sample.items():
                        if k != 'idx':
                            print(f"      {k}: {v}")
        
        print(f"\n[POOL-ADD-MANY] Pool State Change:")
        print(f"  Size: {initial_size} → {final_size} (Δ{final_size - initial_size:+d})")
        print(f"  Capacity: {initial_capacity} → {final_capacity} (Δ{final_capacity - initial_capacity:+d})")
        print(f"  Distribution:")
        print(f"    Low:  {initial_low} → {final_low} (Δ{final_low - initial_low:+d})")
        print(f"    High: {initial_high} → {final_high} (Δ{final_high - initial_high:+d})")
        print(f"    Cold: {initial_cold} → {final_cold} (Δ{final_cold - initial_cold:+d})")
        
        # Verification check
        actual_delta = final_size - initial_size
        if actual_delta != added_count:
            print(f"\n[POOL-ADD-MANY] ⚠️  WARNING: Discrepancy detected!")
            print(f"    Expected to add: {added_count}")
            print(f"    Actual size delta: {actual_delta}")
            print(f"    Difference: {added_count - actual_delta}")
            if actual_delta < added_count:
                print(f"    Likely cause: Items were evicted during rebalancing")
        
        print(f"{'='*70}\n")
        
        if added_count < len(items):
            logger.debug(f"Added {added_count}/{len(items)} items (rejected: {rejected_validation + rejected_full})")

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
        
        
        if chosen:
            low_count = sum(1 for i in chosen_idx if i < len(origins) and origins[i] == "low")
            high_count = len(chosen_idx) - low_count
            
            chosen_rewards = [r.reward for r in chosen if r.reward is not None]
            print(f"[SAMPLING-DEBUG] Sampled {len(chosen)} medium items: {low_count} from low, {high_count} from high")
            if chosen_rewards:
                print(f"  Reward: mean={np.mean(chosen_rewards):.3f}, range=[{np.min(chosen_rewards):.3f}, {np.max(chosen_rewards):.3f}]")
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
        from collections import Counter
        pop_counts = Counter()
        candidates: List[QueryRecord] = []
        origins: List[str] = []
        want_high = True

        while take > 0 and (self._low_heap or self._active_high):
            pulled = False

            if want_high and self._active_high:
                popped = self._pop_high_min_unlocked()
                if popped is not None:
                    pop_counts['high'] += 1  # Track source
                    _, _, it = popped
                    candidates.append(it)
                    origins.append("high")
                    take -= 1
                    pulled = True
            elif (not want_high) and self._low_heap:
                _, _, it = heapq.heappop(self._low_heap)
                pop_counts['low'] += 1  # Track source
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

        if candidates:
            print(f"[POP-DEBUG] Popped {len(candidates)} candidates: {dict(pop_counts)}")
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
    
    def _remove_high_heap_max_unlocked(self) -> bool:
        """Remove item with largest reward from high heap (best high-reward item)."""
        self._clean_high_max_top_unlocked()
        while self._high_heap_max:
            neg_r, seq, evicted_rec = heapq.heappop(self._high_heap_max)  # ← Changed: capture record
            if seq in self._active_high:
                self._active_high.remove(seq)
                self._total_evicted += 1
                
                # Clean up record index to prevent memory leak
                self._cleanup_record_index(evicted_rec)
                
                return True
        return False

    def _insert_when_full_unlocked(self, it: QueryRecord) -> bool:
        """Insert item when pool is full. Always attempts insertion; no 'too-easy' rejection."""
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

        # Evict to maintain capacity (policy unchanged, but no hard drop on insert)
        if not self._remove_low_heap_min_unlocked():
            self._remove_high_heap_min_unlocked()

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

        # First evict worst from low side
        while need > 0 and self._low_heap:
            if self._remove_low_heap_min_unlocked():
                evicted += 1
                need -= 1

        # Then evict worst remaining from high side (high_min)
        while need > 0 and self._active_high:
            if self._remove_high_heap_min_unlocked():
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
        
    def _cleanup_record_index(self, evicted_rec: QueryRecord) -> None:
        """
        Remove evicted record from trainer's _record_index to prevent memory leak.
        
        This is safe to call multiple times for the same record (idempotent).
        Thread-safe via trainer's _inbox_lock.
        """
        if evicted_rec is None:
            return
        
        try:
            # Check if trainer has the record tracking infrastructure
            if not hasattr(self, 'trainer_ref'):
                return
            
            trainer = self.trainer_ref
            if not hasattr(trainer, '_record_index'):
                return
            
            # Use trainer's lock for thread safety (record_index is shared)
            if hasattr(trainer, '_inbox_lock'):
                with trainer._inbox_lock:
                    trainer._record_index.pop(evicted_rec.record_id, None)
            else:
                # Fallback if lock doesn't exist (shouldn't happen in production)
                trainer._record_index.pop(evicted_rec.record_id, None)
                
        except Exception as e:
            pass
        
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
        max_workers: int = 1,
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
        self.max_workers = max(1, int(max_workers))  # Ensure at least 1

        # Metrics (local counters)
        self._processed_count = 0
        self._error_count = 0
        self._api_call_count = 0

        # OpenAI client (consider reading from env instead of hardcoding)
        api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
        self.client = OpenAI(api_key=api_key, timeout=api_timeout)

        # Thread pool for parallel API calls
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

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
        # print(f"\n[CONVERT-FLOW-1] PROTO_TO_QUERY_RECORDS")
        # print(f"  Input batch size: {len(dp.batch.get('input_ids', []))}")
        
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
            try:
                est_val = float(est_list[i]) if est_list[i] is not None else 0.5
                final_reward = np.clip(est_val, -1.0, 1.0) if np.isfinite(est_val) else 0.5
            except (TypeError, ValueError) as e:
                logger.debug(f"Invalid reward value at index {i}: {e}")
                final_reward = 0.5

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

        # print(f"  Conversion stats:")
        # print(f"    Total items: {validation_stats['total']}")
        # print(f"    Unsolvable: {validation_stats['unsolvable']}")
        # print(f"    Extraction failed: {validation_stats['extraction_failed']}")
        # print(f"    Dtype issues: {validation_stats['dtype_issues']}")
        # print(f"    Validation failed: {validation_stats['validation_failed']}")
        # print(f"    Success: {validation_stats['success']}")
        # print(f"[CONVERT-FLOW-1] END\n")
        
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

        # ---- Refresh mode (string or bool for backward compat) ----
        mode_raw = getattr(self.config.dynamic_data, "refresh_reward", False)
        if mode_raw is True:
            self.refresh_mode = "children_aggregation"
        elif mode_raw in {"children_aggregation", "path_aggregation"}:
            self.refresh_mode = mode_raw
        else:
            self.refresh_mode = None  # disabled

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
        
        # Initialize origin analysis logger
        self.origin_analysis_logger = OriginAnalysisLogger(
            log_dir, experiment_name,
            buffer_size=getattr(config.trainer, 'origin_analysis_buffer_size', 500)
        )
        
        # Initialize efficiency analyzer
        self.efficiency_analyzer = EfficiencyAnalyzer(
            log_dir, experiment_name,
            model_config=config,
            buffer_size=getattr(config.trainer, 'efficiency_buffer_size', 200)
        )

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
            # 1. Extract all records from pool (with lock)
            with self.query_pool._lock:
                low_records = [rec for _, _, rec in self.query_pool._low_heap]
                high_records = [rec for _, seq, rec in self.query_pool._high_heap_min 
                            if seq in self.query_pool._active_high]
                cold_records = list(self.query_pool._cold_queue)
                
                pool_metrics = {
                    'total_added': self.query_pool._total_added,
                    'total_sampled': self.query_pool._total_sampled,
                    'total_evicted': self.query_pool._total_evicted,
                    'seq': next(self.query_pool._seq),  # Save counter state
                }
            
            logger.info(f"  Extracted {len(low_records)} low, {len(high_records)} high, {len(cold_records)} cold records")
            
            # 2. Collect inbox (thread-safe)
            with self._inbox_lock:
                inbox_protos = list(self._annotated_inbox)
            
            logger.info(f"  Extracted {len(inbox_protos)} inbox protos")
            
            # 3. Serialize all records (CPU + validation)
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
            
            logger.info("  Serializing pool records...")
            low_serialized = _serialize_records_safe(low_records, "low")
            high_serialized = _serialize_records_safe(high_records, "high")
            cold_serialized = _serialize_records_safe(cold_records, "cold")
            
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
                    'low_records': low_serialized,
                    'high_records': high_serialized,
                    'cold_records': cold_serialized,
                    'max_size': self.query_pool._max_size,
                    'low_fraction': self.query_pool._low_fraction,
                    'mixed_easy_medium': getattr(self.query_pool, '_mixed_easy_medium', False),
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
            logger.info(f"  Pool: {len(low_serialized)} low, {len(high_serialized)} high, {len(cold_serialized)} cold")
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
        Load training state for crash recovery with detailed logging.
        Returns True if successful, False if no state found.
        """
        import gzip
        import time
        from collections import Counter
        
        state_path = os.path.join(checkpoint_dir, "training_state.pt.gz")
        
        if not os.path.exists(state_path):
            logger.info(f"No training state found at {state_path}, starting fresh")
            return False
        
        # File info
        file_size_mb = os.path.getsize(state_path) / (1024 * 1024)
        print(f"\n{'='*70}")
        print(f"LOADING TRAINING STATE")
        print(f"{'='*70}")
        print(f"Source: {state_path}")
        print(f"File size: {file_size_mb:.1f} MB")
        
        load_start = time.time()
        
        try:
            # ========== STAGE 1: Load File ==========
            print(f"\n[STAGE 1] Loading compressed state file...")
            with gzip.open(state_path, 'rb') as f:
                state = torch.load(f, map_location='cpu', weights_only=False)
            
            version = state.get('version', 1)
            save_timestamp = state.get('save_timestamp', 'unknown')
            if isinstance(save_timestamp, (int, float)):
                from datetime import datetime
                save_time_str = datetime.fromtimestamp(save_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            else:
                save_time_str = str(save_timestamp)
            
            print(f"  ✓ State version: {version}")
            print(f"  ✓ Saved at: {save_time_str}")
            
            if version > 2:
                raise ValueError(f"Unsupported state version {version} (current version: 2)")
            
            # ========== STAGE 2: Restore Counters ==========
            print(f"\n[STAGE 2] Restoring training counters...")
            self.global_steps = state['global_steps']
            self.gen_steps = state.get('gen_steps', self.global_steps)
            self.current_epoch = state['current_epoch']
            self._reseed_round = state['reseed_round']
            
            print(f"  ✓ Global steps: {self.global_steps}")
            print(f"  ✓ Generation steps: {self.gen_steps}")
            print(f"  ✓ Current epoch: {self.current_epoch}")
            print(f"  ✓ Reseed round: {self._reseed_round}")
            
            # ========== STAGE 3: Deserialize Pool Records ==========
            print(f"\n[STAGE 3] Deserializing query pool records...")
            
            def _deserialize_records_safe(serialized_list, label):
                records = []
                failed = 0
                corruption_reasons = Counter()
                
                print(f"  Processing {len(serialized_list)} {label} records...")
                
                for i, ser in enumerate(serialized_list):
                    try:
                        rec = _deserialize_query_record(ser)
                        # Validate after deserializing
                        if not self._validate_and_fix_record(rec):
                            logger.warning(f"    Record {i}: Validation failed")
                            corruption_reasons['validation_failed'] += 1
                            failed += 1
                            continue
                        records.append(rec)
                    except Exception as e:
                        error_type = type(e).__name__
                        corruption_reasons[error_type] += 1
                        if failed < 3:  # Show first 3 errors in detail
                            logger.warning(f"    Record {i}: {error_type}: {e}")
                        failed += 1
                
                print(f"    ✓ Loaded: {len(records)}/{len(serialized_list)} records")
                if failed > 0:
                    logger.warning(f"    ✗ Failed: {failed} records")
                    logger.warning(f"      Reasons: {dict(corruption_reasons)}")
                
                return records
            
            pool_state = state['query_pool']
            
            low_records = _deserialize_records_safe(pool_state['low_records'], "LOW heap")
            high_records = _deserialize_records_safe(pool_state['high_records'], "HIGH heap")
            cold_records = _deserialize_records_safe(pool_state['cold_records'], "COLD queue")
            
            # Analyze loaded pool records
            all_pool_records = low_records + high_records + cold_records
            print(f"\n  Pool Records Summary:")
            print(f"    Total loaded: {len(all_pool_records)}")
            print(f"      Low heap:   {len(low_records)}")
            print(f"      High heap:  {len(high_records)}")
            print(f"      Cold queue: {len(cold_records)}")
            
            # Reward distribution
            if all_pool_records:
                rewards = [r.reward for r in all_pool_records if r.reward is not None and np.isfinite(r.reward)]
                if rewards:
                    print(f"    Reward stats:")
                    print(f"      Valid: {len(rewards)}/{len(all_pool_records)}")
                    print(f"      Mean: {np.mean(rewards):.3f}")
                    print(f"      Std:  {np.std(rewards):.3f}")
                    print(f"      Min:  {np.min(rewards):.3f}")
                    print(f"      Max:  {np.max(rewards):.3f}")
                    print(f"      Median: {np.median(rewards):.3f}")
            
            # Origin distribution
            if all_pool_records:
                origins = [(r.meta or {}).get('origin', 'unknown') for r in all_pool_records]
                origin_counts = Counter(origins)
                print(f"    Origin distribution:")
                for origin, count in origin_counts.most_common():
                    pct = count / len(all_pool_records) * 100
                    print(f"      {origin}: {count} ({pct:.1f}%)")
            
            # ========== STAGE 4: Recreate Query Pool ==========
            print(f"\n[STAGE 4] Recreating query pool...")
            print(f"  Config:")
            print(f"    max_size: {pool_state['max_size']}")
            print(f"    low_fraction: {pool_state['low_fraction']}")
            print(f"    mixed_easy_medium: {pool_state.get('mixed_easy_medium', False)}")
            
            self.query_pool = ThreadSafeQueryPool(
                max_size=pool_state['max_size'],
                low_fraction=pool_state['low_fraction'],
                rng=self._rng,
                mixed_easy_medium=pool_state.get('mixed_easy_medium', False),
                trainer_ref=self,
            )
            
            # Restore sequence counter
            if 'seq' in pool_state['metrics']:
                import itertools
                seq_val = pool_state['metrics']['seq']
                self.query_pool._seq = itertools.count(seq_val)
                print(f"  ✓ Sequence counter: {seq_val}")
            
            # Re-add all records
            if all_pool_records:
                print(f"  Adding {len(all_pool_records)} records to pool...")
                pool_size_before = self.query_pool.size()
                self.query_pool.add_many(all_pool_records)
                pool_size_after = self.query_pool.size()
                actual_added = pool_size_after - pool_size_before
                
                print(f"    Pool size: {pool_size_before} → {pool_size_after}")
                print(f"    Actually added: {actual_added}/{len(all_pool_records)}")
                
                if actual_added < len(all_pool_records):
                    rejected = len(all_pool_records) - actual_added
                    logger.warning(f"    ⚠ {rejected} records were rejected/evicted during re-heapification")
            
            # Restore pool metrics
            self.query_pool._total_added = pool_state['metrics']['total_added']
            self.query_pool._total_sampled = pool_state['metrics']['total_sampled']
            self.query_pool._total_evicted = pool_state['metrics']['total_evicted']
            
            print(f"  ✓ Pool lifetime metrics:")
            print(f"    Total added:   {self.query_pool._total_added}")
            print(f"    Total sampled: {self.query_pool._total_sampled}")
            print(f"    Total evicted: {self.query_pool._total_evicted}")
            
            # ========== STAGE 5: Restore Archives ==========
            print(f"\n[STAGE 5] Restoring trained archive...")
            archive_serialized = state['trained_archive']
            print(f"  Serialized archive size: {len(archive_serialized)}")
            
            self._trained_archive = {}
            archive_failed = 0
            for k, ser in archive_serialized.items():
                try:
                    rec = _deserialize_query_record(ser)
                    if self._validate_and_fix_record(rec):
                        self._trained_archive[k] = rec
                    else:
                        archive_failed += 1
                except Exception as e:
                    archive_failed += 1
                    if archive_failed <= 3:
                        logger.debug(f"  Failed to load archive record {k}: {e}")
            
            print(f"  ✓ Loaded: {len(self._trained_archive)}/{len(archive_serialized)} archive records")
            if archive_failed > 0:
                logger.warning(f"  ✗ Failed: {archive_failed} archive records")
            
            # Archive stats
            if self._trained_archive:
                archive_rewards = [r.reward for r in self._trained_archive.values() 
                                if r.reward is not None and np.isfinite(r.reward)]
                if archive_rewards:
                    print(f"  Archive reward stats:")
                    print(f"    Mean: {np.mean(archive_rewards):.3f}")
                    print(f"    Range: [{np.min(archive_rewards):.3f}, {np.max(archive_rewards):.3f}]")
            
            # ========== STAGE 6: Restore Record Index ==========
            print(f"\n[STAGE 6] Restoring record index...")
            index_serialized = state['record_index']
            print(f"  Serialized index size: {len(index_serialized)}")
            
            self._record_index = {}
            index_failed = 0
            for k, ser in index_serialized.items():
                try:
                    rec = _deserialize_query_record(ser)
                    if self._validate_and_fix_record(rec):
                        self._record_index[k] = rec
                    else:
                        index_failed += 1
                except Exception as e:
                    index_failed += 1
                    if index_failed <= 3:
                        logger.debug(f"  Failed to load index record {k}: {e}")
            
            print(f"  ✓ Loaded: {len(self._record_index)}/{len(index_serialized)} index records")
            if index_failed > 0:
                logger.warning(f"  ✗ Failed: {index_failed} index records")
            
            # ========== STAGE 7: Restore Lineage ==========
            print(f"\n[STAGE 7] Restoring lineage graphs...")
            lineage = state['lineage']
            
            self._parent_to_children = defaultdict(set, {
                k: set(v) for k, v in lineage['parent_to_children'].items()
            })
            self._child_to_parent = lineage['child_to_parent']
            
            print(f"  ✓ Parent→Children map: {len(self._parent_to_children)} parents")
            print(f"  ✓ Child→Parent map: {len(self._child_to_parent)} children")
            
            # Lineage stats
            if self._parent_to_children:
                children_counts = [len(children) for children in self._parent_to_children.values()]
                print(f"  Lineage stats:")
                print(f"    Max children per parent: {max(children_counts)}")
                print(f"    Avg children per parent: {np.mean(children_counts):.1f}")
                print(f"    Total parent-child links: {sum(children_counts)}")
            
            # ========== STAGE 8: Restore Seed Template ==========
            print(f"\n[STAGE 8] Restoring seed template...")
            seed_serialized = state['seed_records_template']
            print(f"  Serialized seed template size: {len(seed_serialized)}")
            
            self._seed_records_template = []
            seed_failed = 0
            for i, ser in enumerate(seed_serialized):
                try:
                    rec = _deserialize_query_record(ser)
                    if self._validate_and_fix_record(rec):
                        self._seed_records_template.append(rec)
                    else:
                        seed_failed += 1
                except Exception as e:
                    seed_failed += 1
                    if seed_failed <= 3:
                        logger.warning(f"  Failed to load seed record {i}: {e}")
            
            print(f"  ✓ Loaded: {len(self._seed_records_template)}/{len(seed_serialized)} seed records")
            if seed_failed > 0:
                logger.warning(f"  ✗ Failed: {seed_failed} seed records")
            
            # ========== STAGE 9: Restore Inbox ==========
            print(f"\n[STAGE 9] Restoring teacher inbox...")
            inbox_serialized = state['inbox_protos']
            print(f"  Serialized inbox size: {len(inbox_serialized)} protos")
            
            with self._inbox_lock:
                self._annotated_inbox.clear()
                inbox_loaded = 0
                inbox_failed = 0
                total_items = 0
                
                for i, ser in enumerate(inbox_serialized):
                    try:
                        dp = _deserialize_dataproto(ser)
                        if dp is not None:
                            self._annotated_inbox.append(dp)
                            inbox_loaded += 1
                            total_items += len(dp.batch.get('input_ids', []))
                        else:
                            inbox_failed += 1
                    except Exception as e:
                        inbox_failed += 1
                        if inbox_failed <= 3:
                            logger.warning(f"  Failed to load inbox proto {i}: {e}")
            
            print(f"  ✓ Loaded: {inbox_loaded}/{len(inbox_serialized)} protos")
            print(f"  ✓ Total items in inbox: {total_items}")
            if inbox_failed > 0:
                logger.warning(f"  ✗ Failed: {inbox_failed} protos")
            
            # ========== STAGE 10: Restore Metrics ==========
            print(f"\n[STAGE 10] Restoring augmentation metrics...")
            self._augmentation_metrics = state['augmentation_metrics']
            
            print(f"  Augmentation metrics:")
            for key, value in self._augmentation_metrics.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
            
            # ========== FINAL SUMMARY ==========
            load_time = time.time() - load_start
            
            print(f"\n{'='*70}")
            print(f"TRAINING STATE LOADED SUCCESSFULLY")
            print(f"{'='*70}")
            print(f"Load time: {load_time:.1f}s")
            print(f"")
            print(f"Training Status:")
            print(f"  Resuming from step: {self.global_steps}")
            print(f"  Current epoch: {self.current_epoch}")
            print(f"  Generation steps: {self.gen_steps}")
            print(f"")
            print(f"Query Pool:")
            print(f"  Current size: {self.query_pool.size()}")
            print(f"  Capacity: {self.query_pool._max_size}")
            print(f"  Distribution:")
            print(f"    └─ Low:  {len(self.query_pool._low_heap)}")
            print(f"    └─ High: {self.query_pool._high_size_unlocked()}")
            print(f"    └─ Cold: {len(self.query_pool._cold_queue)}")
            print(f"")
            print(f"Data Structures:")
            print(f"  Trained archive: {len(self._trained_archive)} items")
            print(f"  Record index: {len(self._record_index)} items")
            print(f"  Lineage:")
            print(f"    └─ Parents: {len(self._parent_to_children)}")
            print(f"    └─ Children: {len(self._child_to_parent)}")
            print(f"  Seed template: {len(self._seed_records_template)} items")
            print(f"  Teacher inbox: {len(self._annotated_inbox)} protos pending")
            print(f"{'='*70}\n")
            return True
            
        except Exception as e:
            logger.error(f"\n{'='*70}")
            logger.error(f"FAILED TO LOAD TRAINING STATE")
            logger.error(f"{'='*70}")
            logger.error(f"Error: {e}")
            logger.error(f"Checkpoint: {state_path}")
            logger.error(f"{'='*70}\n")
            import traceback
            traceback.print_exc()
            return False
 
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
            
            response_mask = batch.batch.get("response_mask")
            attention_mask = batch.batch.get("attention_mask")
            
            if response_mask is None:
                print("Cannot save rollouts: no response_mask found")
                return
            
            # Ensure response_mask matches sequence length (handle T vs T-1 mismatch)
            if response_mask.size(-1) != sequences.size(-1):
                if response_mask.size(-1) == sequences.size(-1) - 1:
                    # Pad response_mask with a 0 at the beginning (query position)
                    response_mask = torch.cat([
                        torch.zeros_like(response_mask[..., :1]),
                        response_mask
                    ], dim=-1)
                else:
                    # Trim to common length
                    min_len = min(response_mask.size(-1), sequences.size(-1))
                    response_mask = response_mask[..., :min_len]
                    sequences = sequences[..., :min_len]
                    if attention_mask is not None:
                        attention_mask = attention_mask[..., :min_len]
            
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
                    # Get tensors for this item
                    seq = sequences[i].detach().cpu()
                    resp_mask = response_mask[i].detach().cpu()
                    attn = attention_mask[i].detach().cpu() if attention_mask is not None else None
                    
                    # Filter out padding using attention mask
                    if attn is not None:
                        valid_mask = (attn > 0)
                        seq = seq[valid_mask]
                        resp_mask = resp_mask[valid_mask]
                    
                    if seq.numel() == 0:
                        continue  # Skip empty sequences
                    
                    # Separate query and rollout using response_mask
                    # Query: where response_mask == 0
                    # Rollout: where response_mask > 0
                    query_mask = (resp_mask == 0)
                    rollout_mask = (resp_mask > 0)
                    
                    query_ids = seq[query_mask].tolist()
                    rollout_ids = seq[rollout_mask].tolist()
                    
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
            valid_rewards = [r["reward"] for r in records if r["reward"] is not None]
            
            logger.info(f"[ROLLOUT-SAVE] Saved {len(records)} rollout records to {output_file}")
            logger.info(f"  Rollout length: min={min(rollout_lengths)}, max={max(rollout_lengths)}, "
                    f"mean={np.mean(rollout_lengths):.1f}, median={np.median(rollout_lengths):.1f}")
            
            if valid_rewards:
                print(f"  Reward stats: min={min(valid_rewards):.3f}, max={max(valid_rewards):.3f}, "
                        f"mean={np.mean(valid_rewards):.3f}")
        except Exception as e:
            # Never crash training due to debug logging
            print(f"Failed to save rollout records at step {step}: {e}")

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

    def _is_acyclic(self, p2c: dict[str, set[str]]) -> bool:
        seen, stack = set(), set()
        def dfs(u):
            if u in stack: return False
            if u in seen: return True
            seen.add(u); stack.add(u)
            for v in p2c.get(u, ()):
                if not dfs(v): return False
            stack.remove(u); return True
        return all(dfs(u) for u in p2c.keys())
        
    def _topo_from_leaves(self, p2c, nodes):
        # reverse edges for leaf→root pass
        c2p = {u:set() for u in nodes}
        for p, cs in p2c.items():
            for c in cs: c2p.setdefault(c, set()).add(p)
        from collections import deque
        q = deque([u for u in nodes if not p2c.get(u)])  # leaves
        order, seen = [], set()
        while q:
            u = q.popleft()
            if u in seen: continue
            seen.add(u); order.append(u)
            for p in c2p.get(u, ()):
                # push parent only after all its children are seen
                if all((gc in seen) for gc in p2c.get(p, ())):
                    q.append(p)
        # If some nodes remain (isolated/cyclic), append them anyway
        order += [u for u in nodes if u not in seen]
        return order

    def _build_levels(self):
        parent_to_children, records = self._snapshot_lineage_and_records()
        nodes = set(records.keys()) | set(parent_to_children.keys())
        for cs in parent_to_children.values():
            nodes.update(cs)

        depth = {u: 0 for u in nodes}

        # Kahn-style: track in-degrees to detect cycles
        indeg = {u: 0 for u in nodes}
        for u, cs in parent_to_children.items():
            for v in cs:
                indeg[v] = indeg.get(v, 0) + 1
        
        # Guard: if there is a cycle, fall back to bounded relaxation and surface a metric
        is_dag = self._is_acyclic(parent_to_children)
        if any(in_deg > 0 for u, in_deg in indeg.items()) and not is_dag:
            try:
                self._augmentation_metrics["lineage/cycle_detected"] = 1
            except Exception:
                pass
            max_iters = 4 * max(1, len(nodes))
            iters = 0
            changed = True
            while changed and iters < max_iters:
                changed, iters = False, iters + 1
                for u in nodes:
                    cs = parent_to_children.get(u, ())
                    if not cs:
                        continue
                    m = max(depth.get(c, 0) for c in cs) + 1
                    if m != depth.get(u, 0):
                        depth[u] = m
                        changed = True
        else:
            # Acyclic: do a single bottom-up DP
            topo = self._topo_from_leaves(parent_to_children, nodes)
            for u in topo:
                cs = parent_to_children.get(u, ())
                if cs:
                    depth[u] = max(depth.get(c, 0) for c in cs) + 1

        levels = {}
        for u, d in depth.items():
            levels.setdefault(d, []).append(u)
            
        # Visibility: parent pointers that don't resolve to any known node
        try:
            dangling = 0
            for p, cs in parent_to_children.items():
                for c in cs:
                    if c not in records:
                        dangling += 1
            self._augmentation_metrics["lineage/dangling_children"] = dangling
        except Exception:
            pass
        
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
        """Recycle trained items in controlled batches to avoid overwhelming Ray."""
        import math, time, gc
        if not self._trained_archive or self.query_pool is None:
            return 0

        # Levelized refresh
        if self.refresh_mode in {"children_aggregation", "path_aggregation"}:
            refreshed_map = self._refresh_rewards_topo(self.refresh_mode)
        else:
            refreshed_map = {}

        # 1. Make a snapshot copy
        items = list(self._trained_archive.values())   # 17000 -> 20000 -> 19000 -> 20000
        self._rng.shuffle(items)
        
        valid_rewards = [r.reward for r in items if r.reward is not None and np.isfinite(r.reward)]
        if valid_rewards:
            sorted_rewards = sorted(valid_rewards)
            n = len(sorted_rewards)
            
            print(f"\n[DISTRIBUTION-DEBUG] Archive reward distribution:")
            print(f"  Total: {n} items")
            print(f"  Mean: {np.mean(sorted_rewards):.3f}")
            print(f"  Median: {sorted_rewards[n//2]:.3f}")  # ← THE KEY NUMBER!
            print(f"  Quartiles: Q1={sorted_rewards[n//4]:.3f}, Q3={sorted_rewards[3*n//4]:.3f}")
            print(f"  Percentiles:")
            for p in [10, 25, 50, 75, 90]:
                idx = int(n * p / 100)
                print(f"    P{p}: {sorted_rewards[idx]:.3f}")
        
        print(f"\n[ARCHIVE-DEBUG] Before reinsertion:")
        print(f"  Archive size: {len(items)}")

        origins = [(r.meta or {}).get("origin", "unknown") for r in items[:100]]  # Sample first 100
        from collections import Counter
        print(f"  Sample origins: {dict(Counter(origins))}")

        rewards_sample = [r.reward for r in items[:100]]
        print(f"  Sample rewards: mean={np.nanmean(rewards_sample):.3f}, "
            f"min={np.nanmin(rewards_sample):.3f}, max={np.nanmax(rewards_sample):.3f}")

        trained_rounds = [(r.meta or {}).get("trained_round", -1) for r in items[:100]]
        print(f"  Sample trained_rounds: min={min(trained_rounds)}, max={max(trained_rounds)}, "
            f"current_step={self.global_steps}")
        
        # 2. Clear immediately to prevent duplicates on next reinsertion
        self._trained_archive.clear()
        
        print(f"[dynamic] Cleared archive after copying {len(items)} items for reinsertion")

        # 3. Now process the snapshot (archive is already empty)
        BATCH_SIZE = 512
        total_added = 0
        
        for batch_start in range(0, len(items), BATCH_SIZE):
            batch_items = items[batch_start:batch_start + BATCH_SIZE]
            to_add = []
            
            corrupted_count = 0
            copy_failed_count = 0
            validation_failed_count = 0

            for idx, base in enumerate(batch_items):
                # CRITICAL FIX: Deep copy BEFORE validation (don't modify original)
                try:
                    rec = QueryRecord(
                        raw_prompt_data=base.raw_prompt_data.copy() if isinstance(base.raw_prompt_data, np.ndarray) else base.raw_prompt_data,
                        input_ids=base.input_ids.clone() if base.input_ids is not None else None,
                        attention_mask=base.attention_mask.clone() if base.attention_mask is not None else None,
                        position_ids=base.position_ids.clone() if base.position_ids is not None else None,
                        gt=base.gt,
                        reward=base.reward,
                        est_reward=base.est_reward,
                        meta=dict(base.meta or {}),
                        record_id=base.record_id,
                        original_text=base.original_text,
                        augmented_text=base.augmented_text,
                        teacher_response=base.teacher_response,
                        creation_time=base.creation_time,
                    )
                except Exception as e:
                    print(f"[COPY-FAIL-DEBUG] Record copy failed at batch {batch_start//BATCH_SIZE}, item {idx}")
                    print(f"  raw_prompt_data type: {type(base.raw_prompt_data)}")
                    print(f"  raw_prompt_data shape: {base.raw_prompt_data.shape if hasattr(base.raw_prompt_data, 'shape') else 'N/A'}")
                    print(f"  raw_prompt_data dtype: {base.raw_prompt_data.dtype if hasattr(base.raw_prompt_data, 'dtype') else 'N/A'}")
                    print(f"  input_ids: {'None' if base.input_ids is None else f'tensor shape={base.input_ids.shape}'}")
                    corrupted_count += 1
                    copy_failed_count += 1
                    continue
                
                # Now validate the COPY (base is untouched)
                if not self._validate_and_fix_record(rec):
                    corrupted_count += 1
                    validation_failed_count += 1
                    print(f"[VALIDATION-FAIL-DEBUG] Record validation failed at batch {batch_start//BATCH_SIZE}, item {idx}")
                    print(f"  record_id: {rec.record_id}")
                    print(f"  raw_prompt_data type: {type(rec.raw_prompt_data)}")
                    print(f"  raw_prompt_data shape: {rec.raw_prompt_data.shape if hasattr(rec.raw_prompt_data, 'shape') else 'N/A'}")
                    print(f"  raw_prompt_data dtype: {rec.raw_prompt_data.dtype if hasattr(rec.raw_prompt_data, 'dtype') else 'N/A'}")
                    continue
                
                # Apply refreshed reward if available, else keep original
                nv = refreshed_map.get(rec.record_id, None)
                if nv is not None:
                    rec.reward = nv
                    rec.est_reward = nv
                    
                    # Sync refreshed reward back to index
                    with self._inbox_lock:
                        if rec.record_id in self._record_index:
                            self._record_index[rec.record_id].reward = nv
                            self._record_index[rec.record_id].est_reward = nv
                
                # Sanity check: skip records that somehow lost their reward
                elif rec.reward is None or not np.isfinite(rec.reward):
                    print(f"Skipping record {rec.record_id} with invalid reward during reinsertion")
                    corrupted_count += 1
                    continue
                
                to_add.append(rec)

            if corrupted_count > 0:
                print(f"[REINSERTION-DEBUG] Batch {batch_start//BATCH_SIZE}: "
                    f"corrupted={corrupted_count} (copy_failed={copy_failed_count}, "
                    f"validation_failed={validation_failed_count}), valid={len(to_add)}")

            if to_add:
                self.query_pool.add_many(to_add)
                total_added += len(to_add)
                
                # Yield control to avoid blocking Ray actors
                if batch_start + BATCH_SIZE < len(items):
                    time.sleep(0.1)
                    
        print(f"\n[HEAP-DEBUG] After reinsertion at step {self.global_steps}:")
        print(f"  Pool size: low={len(self.query_pool._low_heap)}, high={self.query_pool._high_size_unlocked()}")

        low_rewards = []
        if self.query_pool._low_heap:
            # Low heap stores (-reward, ...), so negate to get actual rewards
            low_rewards = sorted([-r for r, _, _ in self.query_pool._low_heap])
            print(f"  Low heap (worst to best): [{low_rewards[0]:.3f}, ..., {low_rewards[-1]:.3f}]")
            print(f"  Low heap sample (10 worst): {[f'{x:.2f}' for x in low_rewards[:10]]}")
            print(f"  Low heap sample (10 best):  {[f'{x:.2f}' for x in low_rewards[-10:]]}")

        high_rewards = []
        if self.query_pool._high_heap_min:
            # High heap stores (reward, ...) directly
            high_rewards = sorted([r for r, s, _ in self.query_pool._high_heap_min if s in self.query_pool._active_high])
            if high_rewards:
                print(f"  High heap (worst to best): [{high_rewards[0]:.3f}, ..., {high_rewards[-1]:.3f}]")
                print(f"  High heap sample (10 worst): {[f'{x:.2f}' for x in high_rewards[:10]]}")
                print(f"  High heap sample (10 best):  {[f'{x:.2f}' for x in high_rewards[-10:]]}")

        if low_rewards and high_rewards:
            print(f"  Boundary (low max vs high min): low_max={low_rewards[-1]:.3f}, "
                f"high_min={high_rewards[0]:.3f}")

        if total_added > 5000:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"[dynamic] Reinserted {total_added}/{len(items)} trained queries in {math.ceil(len(items)/BATCH_SIZE)} batches")
        print(f"[dynamic] Archive is now empty with {len(self._trained_archive)} items")
        print(f"[dynamic] Current record index size: {len(self._record_index)} items")
        
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
                            reward=None,            # no initial reward
                            est_reward=None,        # defer
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
        self.tokenizer.padding_side = "left"
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
        
        # print(f"\n{'='*70}")
        # print(f"[AUG-FLOW-1] GENERATE_AUGMENTED_QUERIES START")
        # print(f"{'='*70}")
        # print(f"  num_per_prompt: {num_per_prompt}")
        # print(f"  source_batch size: {len(source_batch.batch.get('input_ids', []))}")
        
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
            # enc = self._tokenize_texts([[{"role": "user", "content": aug_prompt}] for aug_prompt in aug_prompts])
            # enc = self._tokenize_texts([[{"role": "user", "content": aug_prompt}] for aug_prompt in aug_prompts])
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
            diff_factors = [self._parse_difficulty_strict(t, lo=0.75, hi=1.33) for t in diff_texts]
            # print(f"[AUG-DEBUG] Difficulty factors: {diff_factors[:5]} (first 5)", flush=True)
            
            min_len = min(len(base_rewards_for_aug), len(diff_factors), len(new_texts))
            # print(f"[AUG-DEBUG] min_len={min_len} (base_rewards={len(base_rewards_for_aug)}, diff={len(diff_factors)}, texts={len(new_texts)})", flush=True)
            
            est_rewards: List[float] = []

            # Update running average
            if diff_factors:
                n = self._augmentation_metrics.get("total_augmented", 0)
                prev = self._augmentation_metrics.get("avg_difficulty_factor", 1.0)
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

            # print(f"[AUG-DEBUG] Created {len(successful_augmentations)} successful augmentations", flush=True)

            self._augmentation_metrics["total_augmented"] += len(new_texts[:min_len])
            self._augmentation_metrics["augmentation_success_rate"] = (
                min_len / len(aug_prompts)) if aug_prompts else 0.0

            # Tokenize NEW problems
            # print(f"[AUG-DEBUG] Step 8: Tokenizing {min_len} new problems", flush=True)
            # new_enc = self._tokenize_texts([[{"role": "user", "content": new_t}] for new_t in new_texts[:min_len]])
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
                "difficulty_factors": np.array(diff_factors[:min_len], dtype=float),
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
            
            # print(f"\n[AUG-FLOW-1] GENERATE_AUGMENTED_QUERIES END")
            # print(f"  Created {min_len} augmented items")
            # print(f"  Token IDs shape: {result.batch['input_ids'].shape}")
            # print(f"  Estimated rewards: min={np.min(est_rewards):.3f}, max={np.max(est_rewards):.3f}, mean={np.mean(est_rewards):.3f}")
            # print(f"{'='*70}\n")
            
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
        packets: List[DataProto] = []
        with self._inbox_lock:
            inbox_size_before = len(self._annotated_inbox)
            while self._annotated_inbox:
                packets.append(self._annotated_inbox.popleft())
            inbox_size_after = len(self._annotated_inbox)
        
        if not packets:
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
                self.query_pool = ThreadSafeQueryPool(
                    max_size=max_pool_size,
                    trainer_ref=self
                )
                sampling_mode = str(getattr(self.config.dynamic_data, "sampling_mode", "medium_only")).lower()
                self.query_pool.set_mixed_easy_medium(
                    sampling_mode in {"mixed_easy_medium", "mixed", "half"}
                )
                
                # Load seed records from dataloader
                seed_records = self._seed_records_from_loader()
                init_mode = str(getattr(self.config.dynamic_data, "init_mode", "map")).lower()

                if init_mode == "uniform":
                    self.query_pool.initialize_uniform(seed_records)
                else:
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
                if not hasattr(self, 'query_pool') or self.query_pool is None:
                    raise RuntimeError("State loading succeeded but query_pool is None!")
                logger.info(f"  Restored pool has {self.query_pool.size()} items")
                logger.info(f"  Restored archive has {len(self._trained_archive)} items")
            
            # Start teacher annotator (needed regardless of state load)
            try:
                model_name = getattr(self.config.dynamic_data, "teacher_model", "gpt-5-nano")
                teacher_max_workers = int(getattr(self.config.dynamic_data, "teacher_max_workers", 1))
                self.teacher_annotator = AsyncTeacherAnnotator(
                    self,
                    model_name=model_name,
                    augmentation_logger=self.augmentation_logger,
                    immediate_release=True,
                    max_workers=teacher_max_workers, 
                )
                self.teacher_annotator.start()
                logger.info("✓ Teacher annotator thread started")
            except Exception as e:
                logger.error(f"Failed to start teacher annotator: {e}")
                raise
        else:
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
                    step_start_time = time.perf_counter()

                    self._drain_teacher_inbox()

                    is_last_step = (self.global_steps >= self.total_training_steps)

                    if dyn_en and self.global_steps % snapshot_frequency == 0:
                        pool_metrics = self.query_pool.get_metrics()
                        pool_sample = self.query_pool.get_sample_for_logging(20)
                        self.augmentation_logger.log_pool_snapshot(pool_metrics, pool_sample)
                        for k, v in pool_metrics.items():
                            metrics[f"pool/{k}"] = v

                    # Step 2: sample batch from queue
                    if dyn_en:
                        want = self.config.data.train_batch_size

                        if self.query_pool.size() < want:
                            added = self._reinsert_all_trained()
                            print("[REINSERT-DEBUG] Reinserted trained queries:", added)
                            logger.info(f"[dynamic] pool<{want}; recycled {added} trained queries")
                        else:
                            print("[REINSERT-DEBUG] Pool size sufficient, no reinsertion.")

                        with self.efficiency_analyzer.measure_stage(
                            EfficiencyAnalyzer.STAGE_POOL_SAMPLE,
                            self.global_steps,
                            batch_size=want,
                            seq_length=0,
                            new_tokens=0
                        ):
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
                                for idx, reason in invalid_sampled[:5]:
                                    rec = sampled[idx]
                                    print(f"  Item {idx}: {reason}")
                                    print(f"    record_id: {rec.record_id}")
                                    print(f"    origin: {(rec.meta or {}).get('origin', 'unknown')}")
                                    print(f"    trained_round: {(rec.meta or {}).get('trained_round', 'N/A')}")
                                
                                sampled = [rec for i, rec in enumerate(sampled) if i not in [idx for idx, _ in invalid_sampled]]
                                print(f"[POST-SAMPLE-VALIDATION] Removed invalid records, continuing with {len(sampled)} valid items")
                            else:
                                print(f"[POST-SAMPLE-VALIDATION] ✓ All {len(sampled)} records passed validation")
                                
                            rewards_pre_rollout = [r.reward for r in sampled]
                            origins_pre_rollout = [(r.meta or {}).get("origin", "unknown") for r in sampled]
                            
                            from collections import Counter
                            origin_counts = Counter(origins_pre_rollout)
                            
                            origin_reward_stats = {}
                            for origin_type in set(origins_pre_rollout):
                                origin_rewards = [r for r, o in zip(rewards_pre_rollout, origins_pre_rollout) 
                                                if o == origin_type and r is not None]
                                
                                if origin_rewards:
                                    origin_reward_stats[origin_type] = {
                                        "count": len(origin_rewards),
                                        "mean": float(np.mean(origin_rewards)),
                                        "min": float(np.min(origin_rewards)),
                                        "max": float(np.max(origin_rewards)),
                                        "std": float(np.std(origin_rewards)),
                                    }
                                else:
                                    origin_reward_stats[origin_type] = {
                                        "count": len([r for r, o in zip(rewards_pre_rollout, origins_pre_rollout) if o == origin_type]),
                                        "mean": None,
                                        "min": None,
                                        "max": None,
                                        "std": None,
                                        "note": "all_None_rewards",
                                    }
                            
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

                        sampled_records_for_this_step = sampled

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

                        if self.augmentation_logger and self.global_steps % 10 == 0:
                            self.augmentation_logger.log_pool_snapshot(
                                {"event": "batch_sampled", "batch_size": len(sampled)},
                                [r.to_dict() for r in sampled[:5]]
                            )

                        new_batch = self._records_to_dataproto(sampled)
                    else:
                        new_batch: DataProto = self._ensure_dataproto(batch_dict)
                        sampled_records_for_this_step = []

                    num_gen_batches += 1

                    pop_keys = ["input_ids", "attention_mask"]
                    if "position_ids" in new_batch.batch:
                        pop_keys.append("position_ids")

                    _wanted_nt = ("raw_prompt_data", "driver_reward", "record_ids")
                    _nt_to_pop = [k for k in _wanted_nt if k in new_batch.non_tensor_batch]

                    gen_batch = new_batch.pop(
                        batch_keys=pop_keys,
                        non_tensor_batch_keys=_nt_to_pop,
                    )
                    
                    print("[FIT-DEBUG-1] Preparing to generate rollouts...")
                    
                    # ============ TENSOR VALIDATION ============
                    print(f"\n[TENSOR-VALIDATION] Validating {len(gen_batch.batch['input_ids'])} items before generation")

                    corrupted_indices = []
                    device_issues = []
                    shape_issues = []

                    for i in range(len(gen_batch.batch["input_ids"])):
                        try:
                            ids = gen_batch.batch["input_ids"][i]
                            
                            if ids.dim() != 1:
                                shape_issues.append(i)
                                print(f"[TENSOR-VALIDATION] Item {i}: input_ids wrong dim {ids.dim()}, expected 1")
                                corrupted_indices.append(i)
                                continue
                            
                            if ids.numel() == 0:
                                shape_issues.append(i)
                                print(f"[TENSOR-VALIDATION] Item {i}: input_ids is empty")
                                corrupted_indices.append(i)
                                continue
                            
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

                    if corrupted_indices:
                        print(f"\n[TENSOR-VALIDATION] ❌ Found {len(corrupted_indices)} corrupted items:")
                        print(f"  - Shape issues: {len(shape_issues)}")
                        print(f"  - Device issues: {len(device_issues)}")
                        
                        keep_indices = [i for i in range(len(gen_batch.batch["input_ids"])) if i not in corrupted_indices]
                        
                        if not keep_indices:
                            print(f"[TENSOR-VALIDATION] CRITICAL: All items corrupted, skipping this step")
                            continue
                        
                        print(f"[TENSOR-VALIDATION] Keeping {len(keep_indices)}/{len(gen_batch.batch['input_ids'])} valid items")
                        gen_batch = gen_batch[keep_indices]
                        
                    else:
                        print(f"[TENSOR-VALIDATION] ✓ All {len(gen_batch.batch['input_ids'])} items passed validation")

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
                    
                    gen_batch_unrepeated = gen_batch

                    gen_batch = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True
                    )

                    if "position_ids" not in gen_batch.batch and "attention_mask" in gen_batch.batch:
                        gen_batch.batch["position_ids"] = _build_position_ids(gen_batch.batch["attention_mask"])
                        if "position_ids" not in gen_batch_unrepeated.batch:
                            gen_batch_unrepeated.batch["position_ids"] = _build_position_ids(gen_batch_unrepeated.batch["attention_mask"])

                    print(f"[TRAIN-DEBUG] Global step {self.global_steps}: gen_batch size {len(gen_batch.batch.get('input_ids', []))}")
                    print("[FIT-DEBUG-2] Start generating rollouts...")
                    
                    with marked_timer("step", timing_raw):
                        # Step 3: generate rollouts
                        with marked_timer("gen", timing_raw, "red"):
                            gen_seq_len = gen_batch.batch["input_ids"].size(-1)
                            gen_batch_size = gen_batch.batch["input_ids"].size(0)
                            max_new_tokens = getattr(
                                self.config.actor_rollout_ref.rollout, 
                                'response_length', 
                                512
                            )
                            
                            with self.efficiency_analyzer.measure_stage(
                                EfficiencyAnalyzer.STAGE_ROLLOUT_GEN,
                                self.global_steps,
                                batch_size=gen_batch_size,
                                seq_length=gen_seq_len,
                                new_tokens=max_new_tokens
                            ):
                                try:
                                    self._dump_debug_batch(gen_batch_unrepeated, self.global_steps)
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                    timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                                    gen_batch_output.meta_info.pop("timing", None)
                                except Exception as e:
                                    print("="*12+"[SEVERE ERROR]"+"="*12)
                                    print("="*12+"[SEVERE ERROR]"+"="*12)
                                    print("="*12+"[SEVERE ERROR]"+"="*12)
                                    print(f"   Generation failed at step {self.global_steps}: {type(e).__name__}: {str(e)[:200]}")
                                    print(f"   Batch size: {len(gen_batch.batch['input_ids'])}, Max len: {gen_batch.batch['input_ids'].size(-1)}")
                                    print("Skipping this batch to continue training...")
                                    print("="*12+"[SEVERE ERROR]"+"="*12)
                                    print("="*12+"[SEVERE ERROR]"+"="*12)
                                    print("="*12+"[SEVERE ERROR]"+"="*12)
                                    
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                    
                                    metrics["generation/failed_steps"] = metrics.get("generation/failed_steps", 0) + 1
                                    metrics["generation/skip_rate"] = metrics.get("generation/failed_steps", 0) / max(1, self.global_steps)
                                    
                                    batch = None
                                    num_prompt_in_batch = 0
                                    num_gen_batches = 0
                                    raise Exception("Generation failed, aborting training for debugging.")

                        print("[FIT-DEBUG-3] Rollout initial generation completed. Start baseline calculation...")

                        # REMAX baseline
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            with marked_timer("gen_max", timing_raw, "red"):
                                gen_baseline_batch = deepcopy(gen_batch)
                                gen_baseline_batch.meta_info["do_sample"] = False
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                                new_batch = new_batch.union(gen_baseline_output)
                                reward_baseline_tensor = self.reward_fn(new_batch)
                                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                                del gen_baseline_batch, gen_baseline_output

                        print("[FIT-DEBUG-4] Baseline calculation completed. Preparing for reward calculation...")

                        n_rep = int(self.config.actor_rollout_ref.rollout.n)
                        gen_bsz = int(gen_batch.batch["input_ids"].size(0))
                        if gen_bsz % n_rep != 0:
                            print(f"Repeated size {gen_bsz} not divisible by n_rep={n_rep}")
                            continue

                        base_bsz = gen_bsz // n_rep
                        new_batch.non_tensor_batch["uid"] = np.array(
                            [str(uuid.uuid4()) for _ in range(base_bsz)], dtype=object
                        )

                        print("[FIT-DEBUG-5] Preparing repeated batch for reward calculation...")

                        new_batch = new_batch.repeat(repeat_times=n_rep, interleave=True)
                        new_batch = new_batch.union(gen_batch_output)

                        print("[FIT-DEBUG-6] Generation completed. Start reward calculation...")

                        # Step 3 (cont.): reward calc
                        with marked_timer("reward", timing_raw, "yellow"):
                            reward_batch_size = len(new_batch.batch["input_ids"])
                            reward_seq_len = new_batch.batch["input_ids"].size(-1) if new_batch.batch["input_ids"].dim() > 1 else 0
                            
                            with self.efficiency_analyzer.measure_stage(
                                EfficiencyAnalyzer.STAGE_REWARD_COMPUTE,
                                self.global_steps,
                                batch_size=reward_batch_size,
                                seq_length=reward_seq_len,
                                new_tokens=0
                            ):
                                self._prepare_reward_model_inputs(new_batch)

                                rm = list(new_batch.non_tensor_batch.get("reward_model", []))

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
                                    return str(gt).strip() == ""

                                keep = [i for i, d in enumerate(rm) if not _missing_gt(d)]
                                if len(keep) != len(rm):
                                    if not keep:
                                        continue
                                    new_batch = new_batch[keep]

                                n_items = len(new_batch.batch["input_ids"])
                                if "data_source" not in new_batch.non_tensor_batch:
                                    new_batch.non_tensor_batch["data_source"] = np.array(["train"] * n_items, dtype=object)
                                else:
                                    new_batch.non_tensor_batch["data_source"] = _to_indexable_array(
                                        new_batch.non_tensor_batch["data_source"], n_items
                                    )
                                    
                                try:
                                    ds = list(new_batch.non_tensor_batch["data_source"])
                                    new_batch.non_tensor_batch["data_source"] = np.array(
                                        ["math_dapo" if (str(x) == "augment") else (x if x else "train") for x in ds],
                                        dtype=object
                                    )
                                except Exception:
                                    new_batch.non_tensor_batch["data_source"] = np.array(["math_dapo"] * n_items, dtype=object)

                                try:
                                    reward_result = self.reward_fn(new_batch, return_dict=True)
                                    reward_tensor = reward_result["reward_tensor"]
                                    reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                                except Exception as e:
                                    print(f"Error in reward_fn (using fallback): {e}")
                                    reward_tensor = self.reward_fn(new_batch)
                                    reward_extra_infos_dict = {}

                                n_items = len(new_batch.batch["input_ids"])
                                for k, v in reward_extra_infos_dict.items():
                                    arr = _to_indexable_array(v, n_items)
                                    if isinstance(arr, np.ndarray) and arr.ndim != 1:
                                        arr = np.array([arr[i] for i in range(n_items)], dtype=object)
                                    new_batch.non_tensor_batch[k] = arr

                                _sanitize_non_tensor_batch(new_batch)
                                if batch is not None:
                                    _sanitize_non_tensor_batch(batch)

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

                                # Reward update logic
                                try:
                                    traj_rewards = new_batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu().numpy()
                                    uids = list(new_batch.non_tensor_batch["uid"])
                                    uid2rewards = defaultdict(list)
                                    for r, u in zip(traj_rewards, uids):
                                        uid2rewards[u].append(float(r))

                                    base_uid_order = []
                                    seen = set()
                                    for u in uids:
                                        if u not in seen:
                                            base_uid_order.append(u)
                                            seen.add(u)

                                    for i, rec in enumerate(sampled_records_for_this_step):
                                        if i >= len(base_uid_order):
                                            break
                                        u = base_uid_order[i]
                                        if u not in uid2rewards:
                                            continue
                                        avg_r = float(np.mean(uid2rewards[u]))
                                        
                                        archived_rec = QueryRecord(
                                            raw_prompt_data=rec.raw_prompt_data.copy() if isinstance(rec.raw_prompt_data, np.ndarray) else rec.raw_prompt_data,
                                            input_ids=rec.input_ids.clone() if rec.input_ids is not None else None,
                                            attention_mask=rec.attention_mask.clone() if rec.attention_mask is not None else None,
                                            position_ids=rec.position_ids.clone() if rec.position_ids is not None else None,
                                            gt=rec.gt,
                                            reward=avg_r,
                                            est_reward=avg_r,
                                            meta={**(rec.meta or {}), "origin": (rec.meta or {}).get("origin", "seed"), "trained_round": self.global_steps},
                                            record_id=rec.record_id,
                                            original_text=rec.original_text,
                                            augmented_text=rec.augmented_text,
                                            teacher_response=rec.teacher_response,
                                            creation_time=rec.creation_time,
                                        )
                                        
                                        if not self._validate_and_fix_record(archived_rec):
                                            print(f"Skipping archive of corrupted record {rec.record_id}")
                                            continue
                                        
                                        self._trained_archive[rec.record_id] = archived_rec
                                        
                                        with self._inbox_lock:
                                            self._record_index[rec.record_id] = archived_rec

                                    # Reward comparison analysis (simplified)
                                    try:
                                        driver_rewards = pre_rollout_snapshot["cached_rewards"]
                                        est_rewards = pre_rollout_snapshot["est_rewards"]
                                        origins = pre_rollout_snapshot["origins"]
                                        
                                        actual_rewards = [uid2rewards[base_uid_order[i]][0] if i < len(base_uid_order) and base_uid_order[i] in uid2rewards else None 
                                                        for i in range(len(sampled_records_for_this_step))]
                                        
                                        valid_driver = [d for d in driver_rewards if d is not None and np.isfinite(d)]
                                        valid_est = [e for e in est_rewards if e is not None and np.isfinite(e)]
                                        valid_actual = [a for a in actual_rewards if a is not None and np.isfinite(a)]
                                        
                                        driver_actual_pairs = [(d, a) for d, a in zip(driver_rewards, actual_rewards) 
                                                            if d is not None and np.isfinite(d) and a is not None and np.isfinite(a)]
                                        if len(driver_actual_pairs) > 10:
                                            drivers_arr = np.array([p[0] for p in driver_actual_pairs])
                                            actuals_arr = np.array([p[1] for p in driver_actual_pairs])
                                            corr_driver_actual = np.corrcoef(drivers_arr, actuals_arr)[0, 1]
                                            mae_driver_actual = np.mean(np.abs(drivers_arr - actuals_arr))
                                        
                                        est_actual_pairs = [(e, a) for e, a in zip(est_rewards, actual_rewards) 
                                                        if e is not None and np.isfinite(e) and a is not None and np.isfinite(a)]
                                        if len(est_actual_pairs) > 10:
                                            est_arr = np.array([p[0] for p in est_actual_pairs])
                                            actuals_arr = np.array([p[1] for p in est_actual_pairs])
                                            corr_est_actual = np.corrcoef(est_arr, actuals_arr)[0, 1]
                                            mae_est_actual = np.mean(np.abs(est_arr - actuals_arr))
                                        
                                        driver_deltas = [(a - d) for d, a in zip(driver_rewards, actual_rewards) 
                                                        if d is not None and np.isfinite(d) and a is not None and np.isfinite(a)]
                                        if driver_deltas:
                                            improved = sum(1 for d in driver_deltas if d > 0.05)
                                            degraded = sum(1 for d in driver_deltas if d < -0.05)
                                        
                                        est_deltas = [(a - e) for e, a in zip(est_rewards, actual_rewards) 
                                                    if e is not None and np.isfinite(e) and a is not None and np.isfinite(a)]
                                        if est_deltas:
                                            overestimated = sum(1 for d in est_deltas if d < -0.05)
                                            underestimated = sum(1 for d in est_deltas if d > 0.05)

                                    except Exception as e:
                                        print(f"[REWARD-DEBUG] Error in reward comparison: {e}")
                                        
                                    # Log metrics
                                    try:
                                        if 'driver_actual_pairs' in locals() and len(driver_actual_pairs) > 10:
                                            metrics["reward_analysis/cached_actual_correlation"] = corr_driver_actual
                                            metrics["reward_analysis/cached_actual_mae"] = mae_driver_actual
                                            metrics["reward_analysis/cached_actual_pairs"] = len(driver_actual_pairs)

                                        if 'est_actual_pairs' in locals() and len(est_actual_pairs) > 10:
                                            metrics["reward_analysis/est_actual_correlation"] = corr_est_actual
                                            metrics["reward_analysis/est_actual_mae"] = mae_est_actual
                                            metrics["reward_analysis/est_actual_pairs"] = len(est_actual_pairs)
                                        
                                        if 'driver_deltas' in locals() and driver_deltas:
                                            metrics["reward_analysis/cached_delta_mean"] = float(np.mean(driver_deltas))
                                            metrics["reward_analysis/cached_delta_std"] = float(np.std(driver_deltas))
                                            metrics["reward_analysis/items_improved"] = improved
                                            metrics["reward_analysis/items_degraded"] = degraded
                                        
                                        if 'est_deltas' in locals() and est_deltas:
                                            metrics["reward_analysis/est_delta_mean"] = float(np.mean(est_deltas))
                                            metrics["reward_analysis/est_delta_std"] = float(np.std(est_deltas))
                                            metrics["reward_analysis/est_overestimated"] = overestimated
                                            metrics["reward_analysis/est_underestimated"] = underestimated
                                            
                                    except Exception as e:
                                        logger.debug(f"Failed to log reward analysis metrics: {e}")
        
                                    per_traj_avg = []
                                    for u in uids:
                                        vals = uid2rewards.get(u, [])
                                        per_traj_avg.append(float(np.mean(vals)) if vals else np.nan)
                                    new_batch.non_tensor_batch["actual_avg_reward"] = np.asarray(per_traj_avg, dtype=float)

                                    base_reward_overrides_hash = {}
                                    for i, rec in enumerate(sampled_records_for_this_step):
                                        if i >= len(base_uid_order):
                                            break
                                        u = base_uid_order[i]
                                        if u not in uid2rewards:
                                            continue
                                        avg_r = float(np.mean(uid2rewards[u]))
                                        key = self._text_hash(rec.original_text or self._decode_tokens_to_text(rec.raw_prompt_data))
                                        base_reward_overrides_hash[key] = avg_r

                                except Exception as e:
                                    logger.debug(f"attach-actual-reward failed: {e}")
                                    base_reward_overrides_hash = {}

                        print("[FIT-DEBUG-6] Reward calculation completed. Start augmentation...")

                        # Step 4: policy-driven augmentation
                        if do_augment:
                            try:
                                self.query_pool.set_max_size(int(getattr(self.config.dynamic_data, "max_pool_size", 30000)))
                                remain = self.query_pool.capacity_remaining()

                                want_per_prompt = int(aug_cfg.get("num_per_prompt", 1))

                                try:
                                    num_prompts = len(gen_batch_unrepeated.batch["input_ids"])
                                    print("[FIT-DEBUG-7] Pre-augmentation Number of original prompts:", num_prompts)
                                except NameError:
                                    try:
                                        n_rep = int(self.config.actor_rollout_ref.rollout.n)
                                        gen_bsz = int(gen_batch.batch["input_ids"].size(0))
                                        num_prompts = max(1, gen_bsz // max(1, n_rep))
                                    except Exception:
                                        rp = list(gen_batch.non_tensor_batch.get("raw_prompt_data", []))
                                        num_prompts = max(1, len({str(x) for x in rp}))

                                required = want_per_prompt * num_prompts

                                print("[FIT-DEBUG-8] Augmentation capacity check:", {
                                    "want_per_prompt": want_per_prompt,
                                    "num_prompts": num_prompts,
                                    "required_capacity": required,
                                    "capacity_remaining": remain,
                                })
                                
                                if want_per_prompt > 0 and num_prompts > 0 and remain >= required:
                                    aug_cfg_this = dict(aug_cfg or {})
                                    aug_cfg_this["base_reward_overrides_hash"] = base_reward_overrides_hash
                                    
                                    aug_seq_len = gen_batch_unrepeated.batch["input_ids"].size(-1)
                                    aug_batch_size = num_prompts * want_per_prompt
                                    aug_max_tokens = getattr(aug_cfg, "max_new_tokens", 400)
                                    
                                    with self.efficiency_analyzer.measure_stage(
                                        EfficiencyAnalyzer.STAGE_AUGMENT_GEN,
                                        self.global_steps,
                                        batch_size=aug_batch_size,
                                        seq_length=aug_seq_len,
                                        new_tokens=aug_max_tokens
                                    ):
                                        aug_proto = self.generate_augmented_queries(
                                            source_batch=gen_batch_unrepeated,
                                            num_per_prompt=want_per_prompt,
                                            aug_cfg=aug_cfg_this,
                                        )
                                        
                                    aug_size = len(aug_proto.batch["input_ids"])
                                    if self.teacher_annotator is not None and len(aug_proto.batch["input_ids"]) > 0:
                                        if not self.teacher_annotator.enqueue_aug(aug_proto):
                                            metrics["augmentation/queue_full_events"] = metrics.get("augmentation/queue_full_events", 0) + 1
                                else:
                                    metrics["augmentation/skipped_due_to_capacity"] = metrics.get("augmentation/skipped_due_to_capacity", 0) + 1
                                    metrics["augmentation/required_capacity"] = required
                                    metrics["augmentation/capacity_remaining"] = remain
                                    metrics["augmentation/num_prompts"] = num_prompts
                                    metrics["augmentation/num_per_prompt"] = want_per_prompt

                            except Exception as e:
                                print(f"[dynamic] augmentation error: {e}")

                        print("[FIT-DEBUG-9] Augmentation step completed. Start group filtering...")
                        
                        # Group filtering
                        if not self.config.algorithm.filter_groups.enable:
                            batch = new_batch
                        else:
                            metric_name = self.config.algorithm.filter_groups.metric
                            if metric_name == "seq_final_reward":
                                new_batch.non_tensor_batch["seq_final_reward"] = (
                                    new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                                )
                            elif metric_name == "seq_reward":
                                new_batch.non_tensor_batch["seq_reward"] = (
                                    new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                                )

                            prompt_uid2metric_vals = defaultdict(list)
                            prompt_uid2traj_indices = defaultdict(list)
                            
                            for idx, (uid, metric_val) in enumerate(zip(
                                new_batch.non_tensor_batch["uid"],
                                new_batch.non_tensor_batch[metric_name]
                            )):
                                prompt_uid2metric_vals[uid].append(metric_val)
                                prompt_uid2traj_indices[uid].append(idx)

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
                                ka, kb = set(a.batch.keys()), set(b.batch.keys())
                                extra_a, extra_b = ka - kb, kb - ka
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
                                traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                                print(
                                    f"[GROUP-FILTER-DIAG] SUCCESS: Collected {num_prompt_in_batch} prompts. "
                                    f"Trimming batch to {traj_bsz} trajectories and proceeding to PPO update.", flush=True
                                )
                                batch = batch[:traj_bsz]

                        print("[GROUP-FILTER-DIAG] Group filtering completed. Start PPO update...")
                        
                        # Step 7: PPO updating
                        batch.batch["response_mask"] = compute_response_mask(batch)

                        save_freq = self.config.trainer.get("save_rollout_records_freq", 0)
                        if save_freq > 0 and (self.global_steps % save_freq == 0 or is_last_step):
                            self._save_rollout_records(batch, self.global_steps)
                            print("[FIT-DEBUG] Saved rollout records at step", self.global_steps)

                        if self.config.trainer.balance_batch:
                            self._balance_batch(batch, metrics=metrics)

                        batch.meta_info["global_token_num"] = torch.sum(
                            batch.batch["attention_mask"], dim=-1
                        ).tolist()

                        # *** FIX: Define update_batch_size and update_seq_len HERE ***
                        update_batch_size = len(batch.batch["input_ids"])
                        update_seq_len = batch.batch["input_ids"].size(-1) if batch.batch["input_ids"].dim() > 1 else 0

                        with marked_timer("old_log_prob", timing_raw, "blue"):
                            with self.efficiency_analyzer.measure_stage(
                                EfficiencyAnalyzer.STAGE_OLD_LOGPROB,
                                self.global_steps,
                                batch_size=update_batch_size,
                                seq_length=update_seq_len,
                                new_tokens=0
                            ):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]

                            if response_masks.size(-1) != entropys.size(-1):
                                if response_masks.size(-1) == entropys.size(-1) + 1:
                                    response_masks = response_masks[..., 1:]
                                else:
                                    T = min(response_masks.size(-1), entropys.size(-1))
                                    response_masks = response_masks[..., :T]
                                    entropys = entropys[..., :T]

                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=loss_agg_mode,
                            )

                            metrics.update({"actor/entropy": entropy_agg.detach().item()})
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                        if self.use_reference_policy:
                            with marked_timer("ref", timing_raw, "olive"):
                                with self.efficiency_analyzer.measure_stage(
                                    EfficiencyAnalyzer.STAGE_REF_LOGPROB,
                                    self.global_steps,
                                    batch_size=update_batch_size,
                                    seq_length=update_seq_len,
                                    new_tokens=0
                                ):
                                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                    batch = batch.union(ref_log_prob)

                        if self.use_critic:
                            with marked_timer("values", timing_raw, "cyan"):
                                values = self.critic_wg.compute_values(batch)
                                batch = batch.union(values)

                        with marked_timer("adv", timing_raw, "brown"):
                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            with self.efficiency_analyzer.measure_stage(
                                EfficiencyAnalyzer.STAGE_ADVANTAGE,
                                self.global_steps,
                                batch_size=update_batch_size,
                                seq_length=update_seq_len,
                                new_tokens=0
                            ):
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
                                with self.efficiency_analyzer.measure_stage(
                                    EfficiencyAnalyzer.STAGE_CRITIC_UPDATE,
                                    self.global_steps,
                                    batch_size=update_batch_size,
                                    seq_length=update_seq_len,
                                    new_tokens=0
                                ):
                                    critic_output = self.critic_wg.update_critic(batch)
                            metrics.update(reduce_metrics(critic_output.meta_info["metrics"]))

                        if self.config.trainer.critic_warmup <= self.global_steps:
                            with marked_timer("update_actor", timing_raw, "red"):
                                with self.efficiency_analyzer.measure_stage(
                                    EfficiencyAnalyzer.STAGE_ACTOR_UPDATE,
                                    self.global_steps,
                                    batch_size=update_batch_size,
                                    seq_length=update_seq_len,
                                    new_tokens=0
                                ):
                                    actor_output = self.actor_rollout_wg.update_actor(batch)
                            metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    # Log origin analysis for the batch
                    try:
                        self.origin_analysis_logger.log_batch(
                            batch, 
                            self.global_steps,
                            use_critic=self.use_critic
                        )
                    except Exception as e:
                        logger.warning(f"Failed to log origin analysis: {e}")
                    
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
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                    metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    n_gpus = self.resource_pool_manager.get_n_gpus()
                    metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                    timing_raw = defaultdict(float)

                    metrics["train/num_gen_batches"] = num_gen_batches
                    metrics["dynamic/pool_size"] = self.query_pool.size() if dyn_en else 0
                    metrics["dynamic/capacity_remaining"] = self.query_pool.capacity_remaining() if dyn_en else 0

                    if dyn_en:
                        pool_metrics = self.query_pool.get_metrics()
                        for k, v in pool_metrics.items():
                            metrics[f"pool/{k}"] = v

                    for k, v in self._augmentation_metrics.items():
                        metrics[f"augmentation/{k}"] = v

                    if self.teacher_annotator:
                        teacher_metrics = self.teacher_annotator.get_metrics()
                        for k, v in teacher_metrics.items():
                            metrics[f"teacher/{k}"] = v

                    try:
                        is_aug = list(batch.non_tensor_batch.get("is_augmented", []))
                        if is_aug:
                            aug_cnt = int(np.sum(is_aug))
                            total = len(is_aug)
                        else:
                            origins = list(batch.non_tensor_batch.get("origin", []))
                            if origins:
                                aug_cnt = sum(1 for x in origins if str(x) == "augmented")
                                total = len(origins)
                            else:
                                ds = list(batch.non_tensor_batch.get("data_source", []))
                                aug_cnt = sum(1 for x in ds if str(x) == "math_dapo")
                                total = len(ds)
                        metrics["batch/augmented_ratio"] = aug_cnt / max(1, total)
                        metrics["batch/augmented_count"] = aug_cnt
                        metrics["batch/seed_count"] = max(0, total - aug_cnt)
                    except Exception as e:
                        logger.debug(f"augmented ratio calc failed: {e}")

                    try:
                        seq_rewards = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu()
                        metrics["batch/reward_mean"] = float(seq_rewards.mean())
                        metrics["batch/reward_std"] = float(seq_rewards.std(unbiased=False))
                    except Exception:
                        pass

                    try:
                        seq_rewards = batch.batch["token_level_rewards"].sum(dim=-1).detach().cpu()
                        if seq_rewards.numel() == 0:
                            metrics["augmented/reward/count"] = 0
                            metrics["original/reward/count"] = 0
                        else:
                            is_aug_list = list(batch.non_tensor_batch.get("is_augmented", []))
                            if not is_aug_list:
                                origins = list(batch.non_tensor_batch.get("origin", []))
                                if origins:
                                    is_aug_list = [str(x) == "augmented" for x in origins]
                                else:
                                    ds = list(batch.non_tensor_batch.get("data_source", []))
                                    is_aug_list = [str(x) == "math_dapo" for x in ds]

                            L = seq_rewards.numel()
                            if len(is_aug_list) < L:
                                is_aug_list += [False] * (L - len(is_aug_list))
                            elif len(is_aug_list) > L:
                                is_aug_list = is_aug_list[:L]
                            mask = torch.tensor(is_aug_list, dtype=torch.bool)

                            aug, orig = seq_rewards[mask], seq_rewards[~mask]
                            metrics["augmented/reward/count"] = int(aug.numel())
                            metrics["original/reward/count"] = int(orig.numel())
                            if aug.numel():
                                metrics["augmented/reward/mean"] = float(aug.mean())
                                metrics["augmented/reward/min"] = float(aug.min())
                                metrics["augmented/reward/max"] = float(aug.max())
                            if orig.numel():
                                metrics["original/reward/mean"] = float(orig.mean())
                                metrics["original/reward/min"] = float(orig.min())
                                metrics["original/reward/max"] = float(orig.max())
                    except Exception as e:
                        logger.debug(f"Per-origin reward logging failed: {e}")

                    # Origin analysis metrics
                    try:
                        origin_summary = self.origin_analysis_logger.get_summary()
                        for origin in ["seed", "augmented"]:
                            origin_stats = origin_summary.get(origin, {})
                            if origin_stats.get("count", 0) > 0:
                                metrics[f"origin/{origin}/reward_mean"] = origin_stats.get("reward_mean", 0)
                                metrics[f"origin/{origin}/reward_std"] = origin_stats.get("reward_std", 0)
                                metrics[f"origin/{origin}/advantage_mean"] = origin_stats.get("advantage_mean", 0)
                                metrics[f"origin/{origin}/advantage_abs_mean"] = origin_stats.get("advantage_abs_mean", 0)
                                if "accuracy" in origin_stats:
                                    metrics[f"origin/{origin}/accuracy"] = origin_stats["accuracy"]
                                metrics[f"origin/{origin}/count"] = origin_stats["count"]
                        
                        seed_stats = origin_summary.get("seed", {})
                        aug_stats = origin_summary.get("augmented", {})
                        if seed_stats.get("reward_mean") is not None and aug_stats.get("reward_mean") is not None:
                            metrics["origin/delta_reward_mean"] = aug_stats["reward_mean"] - seed_stats["reward_mean"]
                        if seed_stats.get("advantage_mean") is not None and aug_stats.get("advantage_mean") is not None:
                            metrics["origin/delta_advantage_mean"] = aug_stats["advantage_mean"] - seed_stats["advantage_mean"]
                    except Exception as e:
                        logger.debug(f"Failed to compute origin metrics: {e}")

                    # Efficiency metrics
                    try:
                        efficiency_metrics = self.efficiency_analyzer.get_metrics_for_logging()
                        metrics.update(efficiency_metrics)
                    except Exception as e:
                        logger.debug(f"Failed to compute efficiency metrics: {e}")

                    step_total_time = time.perf_counter() - step_start_time
                    metrics["efficiency/total_step_time"] = step_total_time

                    # Save analysis summaries periodically
                    analysis_save_freq = getattr(self.config.trainer, 'analysis_save_freq', 100)
                    if self.global_steps % analysis_save_freq == 0:
                        try:
                            self.origin_analysis_logger.save_summary(self.global_steps)
                            self.origin_analysis_logger.save_distributions(self.global_steps)
                            self.efficiency_analyzer.save_summary(self.global_steps)
                        except Exception as e:
                            logger.warning(f"Failed to save analysis summaries: {e}")

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
            self._cleanup_analysis_loggers()

    def _cleanup_analysis_loggers(self):
        """Cleanup analysis loggers and save final state."""
        # Origin analysis
        origin_logger = getattr(self, "origin_analysis_logger", None)
        if origin_logger is not None:
            try:
                origin_logger.flush_all()
                origin_logger.save_summary(self.global_steps)
                origin_logger.save_distributions(self.global_steps)
                logger.info("Origin analysis logs saved")
            except Exception as e:
                print(f"Error saving origin analysis logs: {e}")
        
        # Efficiency analysis
        efficiency_analyzer = getattr(self, "efficiency_analyzer", None)
        if efficiency_analyzer is not None:
            try:
                efficiency_analyzer.flush_all()
                efficiency_analyzer.save_summary(self.global_steps)
                logger.info("Efficiency analysis logs saved")
            except Exception as e:
                print(f"Error saving efficiency analysis logs: {e}")

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