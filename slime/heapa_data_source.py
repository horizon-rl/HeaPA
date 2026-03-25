"""
heapa_data_source.py - HeaPA DataSource for slime training framework.

Implements the slime DataSource ABC backed by a ThreadSafeQueryPool.

Key flow:
  get_samples(n)        : sample n QueryRecords from pool → list[list[Sample]]
  process_scored_samples: update pool rewards; trigger teacher for medium-difficulty items
  add_samples(partial)  : re-queue aborted/partial samples
  save / load           : checkpoint pool state

Usage (slime CLI):
  --data-source-path slime.heapa_data_source.HeaPADataSource
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from slime.rollout.data_source import DataSource
from slime.utils.processing_utils import load_tokenizer

# Patch slime's read_file to inject 'label' from reward_model['ground_truth']
# for verl-format datasets. Applied here so the patch is active in every process
# that imports this module (including RolloutManager Ray actor subprocesses).
import slime.utils.data as _slime_data_mod
_orig_read_file = _slime_data_mod.read_file


def _patched_read_file(path):
    for row in _orig_read_file(path):
        if not row.get("label"):
            rm = row.get("reward_model")
            if isinstance(rm, dict) and rm.get("ground_truth"):
                row = dict(row)
                row["label"] = str(rm["ground_truth"])
        yield row


_slime_data_mod.read_file = _patched_read_file
from slime.utils.types import Sample

from .heapa_core import QueryRecord, ThreadSafeQueryPool
from .heapa_teacher import SlimeTeacherAnnotator, TeacherAnnotationResult

logger = logging.getLogger(__name__)


class HeaPADataSource(DataSource):
    """
    DataSource that wraps a heap-based query pool with optional LLM augmentation.

    Extra argparse args consumed (all prefixed ``heapa_``):
      --heapa-pool-max-size        Max pool capacity           (default: 30000)
      --heapa-low-fraction         Fraction in low/hard heap   (default: 0.5)
      --heapa-mixed-sampling       Use mixed easy+medium mode  (flag)
      --heapa-teacher-enabled      Enable teacher augmentation (flag)
      --heapa-teacher-model        Teacher model name          (default: gpt-4o-mini)
      --heapa-teacher-workers      Parallel API threads        (default: 4)
      --heapa-teacher-hard-lo      Lower reward threshold for teacher trigger (default: 0.1)
      --heapa-teacher-hard-hi      Upper reward threshold for teacher trigger (default: 0.7)
      --heapa-reseed-threshold     Re-seed pool when size drops below this   (default: 100)
    """

    def __init__(self, args):
        self.args = args
        self._group_index = 0
        self._sample_index = 0

        # ---- Tokenizer ----
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---- Pool ----
        pool_max = int(getattr(args, "heapa_pool_max_size", 30000))
        low_frac = float(getattr(args, "heapa_low_fraction", 0.5))
        self.pool = ThreadSafeQueryPool(max_size=pool_max, low_fraction=low_frac)
        mixed = bool(getattr(args, "heapa_mixed_sampling", False))
        self.pool.set_mixed_easy_medium(mixed)

        # ---- Seed dataset ----
        self._seed_records: List[QueryRecord] = []
        if getattr(args, "rollout_global_dataset", True) and getattr(args, "prompt_data", None):
            self._load_seed_dataset()
        else:
            logger.warning("[HeaPA] No prompt_data specified; pool starts empty.")

        # ---- Teacher ----
        self._teacher_inbox: deque = deque(maxlen=50000)
        self._teacher_inbox_lock = threading.Lock()
        self.teacher: Optional[SlimeTeacherAnnotator] = None
        if getattr(args, "heapa_teacher_enabled", False):
            self.teacher = SlimeTeacherAnnotator(
                data_source=self,
                model_name=getattr(args, "heapa_teacher_model", "gpt-4o-mini"),
                max_workers=int(getattr(args, "heapa_teacher_workers", 4)),
            )
            self.teacher.start()
            logger.info("[HeaPA] Teacher annotator started.")

        # ---- Thresholds ----
        self._hard_lo = float(getattr(args, "heapa_teacher_hard_lo", 0.1))
        self._hard_hi = float(getattr(args, "heapa_teacher_hard_hi", 0.7))
        self._reseed_threshold = int(getattr(args, "heapa_reseed_threshold", 100))

        # ---- Record index: record_id → QueryRecord (for reward updates) ----
        self._record_index: Dict[str, QueryRecord] = {}
        self._record_index_lock = threading.Lock()

        # ---- Trained archive (for pool reseed) ----
        self._trained_archive: Dict[str, QueryRecord] = {}

    # ------------------------------------------------------------------
    # Private: seed loading
    # ------------------------------------------------------------------

    def _load_seed_dataset(self):
        """Load prompt dataset and populate pool with seed QueryRecords."""
        from slime.utils.data import Dataset

        logger.info(f"[HeaPA] Loading seed dataset from {self.args.prompt_data}")
        dataset = Dataset(
            self.args.prompt_data,
            self.tokenizer,
            None,   # processor: text-only model
            self.args.rollout_max_prompt_len,
            prompt_key=getattr(self.args, "input_key", "prompt"),
            label_key=getattr(self.args, "label_key", "label"),
            metadata_key=getattr(self.args, "metadata_key", None),
            apply_chat_template=getattr(self.args, "apply_chat_template", True),
            apply_chat_template_kwargs=getattr(self.args, "apply_chat_template_kwargs", None),
            seed=getattr(self.args, "rollout_seed", 42),
        )

        records = []
        for s in dataset.samples:
            token_ids = list(s.tokens) if s.tokens else self.tokenizer.encode(
                s.prompt if isinstance(s.prompt, str) else "", add_special_tokens=False
            )
            if not token_ids:
                continue
            raw = np.array(token_ids, dtype=np.int64)
            rec = QueryRecord(
                raw_prompt_data=raw,
                gt=s.label,
                meta={"origin": "seed"},
                original_text=s.prompt if isinstance(s.prompt, str) else str(s.prompt),
            )
            records.append(rec)

        self._seed_records = records
        self.pool.initialize_uniform(records)
        logger.info(f"[HeaPA] Pool seeded with {len(records)} records")

    # ------------------------------------------------------------------
    # DataSource ABC
    # ------------------------------------------------------------------

    def get_samples(self, num_samples: int) -> List[List[Sample]]:
        """
        Sample `num_samples` QueryRecords from the pool and convert each to a
        list of n_samples_per_prompt Sample objects.

        Returns: list[list[Sample]]  — outer list = prompts, inner = rollout copies.
        """
        n_per_prompt = int(getattr(self.args, "n_samples_per_prompt", 8))

        # Reseed if pool is getting too small
        if self.pool.size() < self._reseed_threshold and self._seed_records:
            logger.info(f"[HeaPA] Pool size {self.pool.size()} below threshold {self._reseed_threshold}; re-seeding...")
            self._reseed_pool()

        records = self.pool.sample_batch(num_samples)
        if not records:
            logger.warning("[HeaPA] Pool is empty; returning empty sample list.")
            return []

        # Register in record index for later reward update
        with self._record_index_lock:
            for rec in records:
                self._record_index[rec.record_id] = rec

        result: List[List[Sample]] = []
        for rec in records:
            group: List[Sample] = []
            prompt_text = self._tokens_to_prompt(rec)
            label_str = str(rec.gt) if rec.gt is not None else ""

            for _ in range(n_per_prompt):
                s = Sample(
                    prompt=prompt_text,
                    tokens=rec.raw_prompt_data.tolist(),
                    label=label_str,
                    group_index=self._group_index,
                    index=self._sample_index,
                    metadata={
                        "record_id": rec.record_id,
                        "origin": rec.meta.get("origin", "seed"),
                        "rm_type": getattr(self.args, "rm_type", "dapo"),
                    },
                )
                self._sample_index += 1
                group.append(s)
            self._group_index += 1
            result.append(group)

        return result

    def add_samples(self, samples: List[List[Sample]]):
        """
        Add back aborted/partial samples to the pool (if partial_rollout is enabled).

        For partial samples (no reward yet), we push them back as cold items.
        """
        if not samples:
            return
        records = []
        for group in samples:
            # Pick the first sample in the group to represent the prompt
            s = group[0] if group else None
            if s is None:
                continue
            token_ids = s.tokens if s.tokens else []
            if not token_ids:
                continue
            raw = np.array(token_ids, dtype=np.int64)
            rec = QueryRecord(
                raw_prompt_data=raw,
                gt=s.label if s.label else None,
                reward=None,  # unscored; goes to cold queue
                meta={
                    "origin": s.metadata.get("origin", "seed") if s.metadata else "seed",
                    "partial": True,
                    "record_id": s.metadata.get("record_id", str(uuid.uuid4())) if s.metadata else str(uuid.uuid4()),
                },
            )
            records.append(rec)
        if records:
            self.pool.add_many(records)
            logger.debug(f"[HeaPA] Re-queued {len(records)} partial samples")

    def _scalar_reward(self, r) -> float:
        """Extract a scalar float from a reward that may be a dict (e.g. dapo rm returns {'score': ..., 'acc': ...})."""
        if isinstance(r, dict):
            key = getattr(self.args, "reward_key", None) or "score"
            return float(r.get(key, r.get("score", 0.0)))
        return float(r)

    def process_scored_samples(self, groups: List[List[Sample]]):
        """
        Called after rollout completes with scored samples.

        For each group:
          1. Average reward over the group.
          2. Update the corresponding QueryRecord in the pool.
          3. Move record to trained_archive.
          4. If reward is in [hard_lo, hard_hi], submit to teacher for augmentation.
          5. Drain teacher inbox → add new augmented records to pool.
        """
        with self._record_index_lock:
            record_index_snapshot = dict(self._record_index)

        teacher_submissions = 0
        for group in groups:
            if not group:
                continue

            # Average reward
            rewards = [self._scalar_reward(s.reward) for s in group if s.reward is not None]
            if not rewards:
                continue
            avg_reward = sum(rewards) / len(rewards)

            # Find the QueryRecord
            record_id = None
            best_response = ""
            for s in group:
                if s.metadata and "record_id" in s.metadata:
                    record_id = s.metadata["record_id"]
                    # Pick best response (highest reward)
                    if s.reward is not None and (not best_response or self._scalar_reward(s.reward) == max(rewards)):
                        best_response = s.response
                    break

            if record_id is None:
                continue

            rec = record_index_snapshot.get(record_id)
            if rec is None:
                # Try pool update by record_id anyway
                self.pool.update_reward(record_id, avg_reward)
                continue

            # Update reward on the record object
            rec.reward = avg_reward
            self.pool.update_reward(record_id, avg_reward)

            # Archive
            self._trained_archive[record_id] = rec

            # Teacher trigger: medium difficulty items (not too easy, not too hard)
            if (
                self.teacher is not None
                and self._hard_lo < avg_reward < self._hard_hi
                and rec.original_text
                and best_response
            ):
                submitted = self.teacher.submit(record_id, rec.original_text, best_response)
                if submitted:
                    teacher_submissions += 1

        if teacher_submissions > 0:
            logger.info(f"[HeaPA] Submitted {teacher_submissions} queries to teacher for augmentation")

        # Drain teacher inbox
        self._drain_teacher_inbox()

        # Clean up record index to avoid unbounded growth
        with self._record_index_lock:
            if len(self._record_index) > 100000:
                # Keep only records currently in pool
                pool_ids = {r.record_id for r in self.pool.get_all_records()}
                self._record_index = {k: v for k, v in self._record_index.items() if k in pool_ids}

    def _drain_teacher_inbox(self):
        """Convert TeacherAnnotationResults in inbox to QueryRecords and add to pool."""
        new_records: List[QueryRecord] = []
        drained = 0
        while True:
            try:
                result: TeacherAnnotationResult = self._teacher_inbox.popleft()
            except IndexError:
                break
            drained += 1
            if not result.solvable or not result.clean_text:
                continue

            # Tokenize the clean augmented text
            token_ids = self.tokenizer.encode(result.clean_text, add_special_tokens=False)
            if not token_ids:
                continue
            raw = np.array(token_ids, dtype=np.int64)

            rec = QueryRecord(
                raw_prompt_data=raw,
                gt=result.answer,
                est_reward=0.5,  # neutral estimate for new augmented items
                meta={
                    "origin": "augmented",
                    "parent_record_id": result.parent_record_id,
                    "difficulty": result.difficulty,
                },
                original_text=result.original_text,
                augmented_text=result.clean_text,
            )
            new_records.append(rec)

        if new_records:
            self.pool.add_many(new_records)
            logger.info(f"[HeaPA] Integrated {len(new_records)} teacher-augmented records into pool (drained {drained} inbox items)")

    def _reseed_pool(self):
        """Re-add seed records when pool gets too small."""
        if not self._seed_records and not self._trained_archive:
            return
        # Prefer archived records (they've been scored)
        candidates = list(self._trained_archive.values())
        if len(candidates) < self._reseed_threshold:
            candidates += self._seed_records
        import random
        sample_size = min(len(candidates), self._reseed_threshold * 2)
        selected = random.sample(candidates, sample_size)
        fresh_records = []
        for r in selected:
            fresh = QueryRecord(
                raw_prompt_data=r.raw_prompt_data.copy(),
                gt=r.gt,
                reward=None,  # reseed as cold; let them get rescored
                meta={**r.meta, "reseed": True},
                original_text=r.original_text,
            )
            fresh_records.append(fresh)
        self.pool.add_many(fresh_records)
        logger.info(f"[HeaPA] Re-seeded pool with {len(fresh_records)} records")

    def _tokens_to_prompt(self, rec: QueryRecord) -> str:
        """Decode token IDs back to prompt text."""
        if rec.original_text:
            return rec.original_text
        try:
            return self.tokenizer.decode(rec.raw_prompt_data.tolist(), skip_special_tokens=False)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Checkpoint save/load
    # ------------------------------------------------------------------

    def save(self, rollout_id):
        """Save pool state for checkpoint recovery."""
        if not getattr(self.args, "save", None):
            return
        save_dir = os.path.join(self.args.save, "heapa_state")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"heapa_pool_{rollout_id}.pt")

        all_records = self.pool.get_all_records()
        # Serialise records to plain dicts (avoid pickling tensors inside QueryRecord)
        serialized = []
        for rec in all_records:
            serialized.append({
                "raw_prompt_data": rec.raw_prompt_data.tolist(),
                "gt": rec.gt,
                "reward": rec.reward,
                "est_reward": rec.est_reward,
                "meta": rec.meta,
                "record_id": rec.record_id,
                "original_text": rec.original_text,
                "augmented_text": rec.augmented_text,
                "creation_time": rec.creation_time,
            })

        archive_serialized = []
        for rec in self._trained_archive.values():
            archive_serialized.append({
                "raw_prompt_data": rec.raw_prompt_data.tolist(),
                "gt": rec.gt,
                "reward": rec.reward,
                "meta": rec.meta,
                "record_id": rec.record_id,
                "original_text": rec.original_text,
            })

        torch.save({
            "pool_records": serialized,
            "archive_records": archive_serialized,
            "group_index": self._group_index,
            "sample_index": self._sample_index,
            "pool_metrics": self.pool.get_metrics(),
        }, path)
        logger.info(f"[HeaPA] Saved pool state ({len(serialized)} records) to {path}")

    def load(self, rollout_id=None):
        """Load pool state from checkpoint."""
        if not getattr(self.args, "load", None):
            return
        save_dir = os.path.join(self.args.load, "heapa_state")
        if rollout_id is not None:
            path = os.path.join(save_dir, f"heapa_pool_{rollout_id}.pt")
        else:
            # Find latest
            import glob
            pattern = os.path.join(save_dir, "heapa_pool_*.pt")
            files = sorted(glob.glob(pattern))
            if not files:
                logger.info(f"[HeaPA] No pool checkpoint found at {save_dir}")
                return
            path = files[-1]

        if not os.path.exists(path):
            logger.info(f"[HeaPA] Pool checkpoint {path} not found; starting fresh")
            return

        logger.info(f"[HeaPA] Loading pool state from {path}")
        state = torch.load(path, weights_only=False)

        # Restore pool records
        records = []
        for d in state.get("pool_records", []):
            raw = np.array(d["raw_prompt_data"], dtype=np.int64)
            rec = QueryRecord(
                raw_prompt_data=raw,
                gt=d.get("gt"),
                reward=d.get("reward"),
                est_reward=d.get("est_reward"),
                meta=d.get("meta", {}),
                record_id=d.get("record_id", str(uuid.uuid4())),
                original_text=d.get("original_text"),
                augmented_text=d.get("augmented_text"),
            )
            records.append(rec)
        self.pool.restore_from_records(records)

        # Restore trained archive
        self._trained_archive = {}
        for d in state.get("archive_records", []):
            raw = np.array(d["raw_prompt_data"], dtype=np.int64)
            rec = QueryRecord(
                raw_prompt_data=raw,
                gt=d.get("gt"),
                reward=d.get("reward"),
                meta=d.get("meta", {}),
                record_id=d.get("record_id", str(uuid.uuid4())),
                original_text=d.get("original_text"),
            )
            self._trained_archive[rec.record_id] = rec

        self._group_index = state.get("group_index", 0)
        self._sample_index = state.get("sample_index", 0)
        logger.info(
            f"[HeaPA] Loaded pool ({len(records)} records, {len(self._trained_archive)} archive)"
        )
