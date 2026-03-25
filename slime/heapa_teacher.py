"""
heapa_teacher.py - Teacher annotator for HeaPA slime integration.

Adapted from AsyncTeacherAnnotator in dapo_ray_trainer_HeaPA.py.
Uses QueryRecord / text-based API instead of VERL DataProto.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Full, Queue
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Any

import numpy as np

if TYPE_CHECKING:
    from slime.heapa_data_source import HeaPADataSource

logger = logging.getLogger(__name__)


class TeacherAnnotationResult:
    """Result of teacher annotation for a single query."""

    def __init__(
        self,
        parent_record_id: str,
        clean_text: str,
        solvable: bool,
        answer: Optional[str],
        difficulty: float,
        tokenizer_fn: Optional[Callable[[str], List[int]]] = None,
        original_text: Optional[str] = None,
        raw_response: Optional[str] = None,
    ):
        self.parent_record_id = parent_record_id
        self.clean_text = clean_text
        self.solvable = solvable
        self.answer = answer
        self.difficulty = difficulty
        self.tokenizer_fn = tokenizer_fn
        self.original_text = original_text
        self.raw_response = raw_response


class SlimeTeacherAnnotator(threading.Thread):
    """
    Background thread that calls an LLM to augment hard math questions.

    Inputs  : (QueryRecord, generated_response_text) tuples enqueued via submit().
    Outputs : TeacherAnnotationResult objects placed into data_source._teacher_inbox.

    Design:
      - Daemon thread; auto-stops when main thread exits.
      - Bounded input queue with backpressure.
      - Thread pool for parallel API calls (max_workers).
      - JSON-based prompt: asks LLM to extract a clean, harder variant plus answer.
    """

    def __init__(
        self,
        data_source: "HeaPADataSource",
        model_name: str = "gpt-4o-mini",
        api_timeout: float = 30.0,
        max_queue: int = 5000,
        max_workers: int = 4,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        poll_interval: float = 0.2,
    ):
        super().__init__(daemon=True)
        self.data_source = data_source
        self.model_name = model_name
        self.api_timeout = api_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.poll_interval = poll_interval
        self._running = True
        self._shutdown_lock = threading.Lock()

        self.queue: Queue = Queue(maxsize=max_queue)
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # metrics
        self._processed_count = 0
        self._error_count = 0
        self._api_call_count = 0

        api_key = os.getenv("OPENAI_API_KEY", "")
        api_base = os.getenv("OPENAI_API_BASE", None)
        from openai import OpenAI
        client_kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": api_timeout}
        if api_base:
            client_kwargs["base_url"] = api_base
        self.client = OpenAI(**client_kwargs)

    def submit(self, record_id: str, original_text: str, response_text: str) -> bool:
        """Enqueue a (record_id, original_text, response_text) tuple for annotation.

        Returns False if the queue is full (backpressure).
        """
        try:
            self.queue.put_nowait((record_id, original_text, response_text))
            return True
        except Full:
            return False

    def shutdown(self):
        with self._shutdown_lock:
            if not self._running:
                return
            self._running = False
            self.stop_event.set()
            self.executor.shutdown(wait=False)

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "queue_size": self.queue.qsize(),
            "processed_count": self._processed_count,
            "error_count": self._error_count,
            "api_call_count": self._api_call_count,
        }

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def run(self):
        logger.info(f"[Teacher] Started (model={self.model_name}, workers={self.executor._max_workers})")
        while self._running:
            try:
                item = self.queue.get(timeout=self.poll_interval)
            except Empty:
                if self.stop_event.is_set():
                    break
                continue

            if not self._running:
                self.queue.task_done()
                break

            try:
                self.executor.submit(self._annotate_and_deliver, *item)
            except Exception as e:
                logger.warning(f"[Teacher] Failed to submit to thread pool: {e}")
                self._error_count += 1
            finally:
                self.queue.task_done()

    def _annotate_and_deliver(self, record_id: str, original_text: str, response_text: str):
        """Annotate one pair and push result to data_source._teacher_inbox."""
        result = self._call_api_with_retry(record_id, original_text, response_text)
        self._processed_count += 1
        if result is not None and result.solvable and result.clean_text:
            self.data_source._teacher_inbox.append(result)
            logger.debug(f"[Teacher] Delivered annotation for {record_id[:8]}...")
        else:
            logger.debug(f"[Teacher] Discarded annotation for {record_id[:8]}: solvable={getattr(result, 'solvable', None)}")

    # ------------------------------------------------------------------ #
    #  API call helpers
    # ------------------------------------------------------------------ #

    def _make_prompt(self, original: str, generation: str) -> str:
        return (
            "You are a math data cleaner and solver.\n"
            "TASKS:\n"
            "1) Read ORIGINAL and GENERATION. Extract a single, self-contained math problem "
            "statement from GENERATION only. Remove any prefaces, commentary, code fences, "
            "and any 'Answer:' lines. Don't copy text from ORIGINAL. "
            "If no clean question can be extracted, return empty string and mark unsolvable.\n"
            "2) If a clean question exists, decide if it is well-posed. If solvable, compute "
            "ONLY the final numeric answer.\n"
            "3) Estimate relative difficulty vs ORIGINAL on a 0.75-1.33 scale (1.0=same).\n\n"
            "Return ONLY one JSON object on a single line:\n"
            '{"clean":"<string>","solvable":true|false,"answer":"<string or null>","difficulty":<number>}\n'
            "Rules: lowercase true/false/null; 'answer' must be a bare number string when solvable, else null.\n\n"
            f"ORIGINAL:\n{original}\n\nGENERATION:\n{generation}"
        )

    def _call_api_with_retry(
        self,
        record_id: str,
        original_text: str,
        response_text: str,
    ) -> Optional[TeacherAnnotationResult]:
        prompt = self._make_prompt(original_text, response_text)
        for attempt in range(self.max_retries):
            try:
                self._api_call_count += 1
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a careful math data cleaner and solver."},
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=2048,
                    response_format={"type": "json_object"},
                )
                content = ""
                if resp.choices:
                    msg = resp.choices[0].message
                    content = (getattr(msg, "content", "") or "").strip()

                # Strip markdown fences
                content = re.sub(r"^```[a-z]*\s*", "", content.strip())
                content = re.sub(r"\s*```$", "", content)

                obj = json.loads(content)
                clean = (obj.get("clean") or "").strip()
                solvable = bool(obj.get("solvable", False))
                ans = obj.get("answer", None)
                if not solvable:
                    ans = None
                elif isinstance(ans, (int, float)):
                    ans = str(ans)
                diff = float(obj.get("difficulty", 1.0))
                if not np.isfinite(diff):
                    diff = 1.0
                diff = float(np.clip(diff, 0.75, 1.33))

                return TeacherAnnotationResult(
                    parent_record_id=record_id,
                    clean_text=clean,
                    solvable=solvable,
                    answer=ans,
                    difficulty=diff,
                    original_text=original_text,
                    raw_response=content,
                )
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    logger.warning(f"[Teacher] API failed for {record_id[:8]}: {e}")
                    self._error_count += 1
                    return None
        return None
