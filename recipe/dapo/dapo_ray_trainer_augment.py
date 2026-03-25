import uuid
import threading
import numpy as np
import torch
import os
import time

from key import OPENAI_API_KEY
from openai import OpenAI
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

from dataclasses import dataclass
from typing import Optional, List, Dict
from collections import deque
from tqdm import tqdm

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
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


# ==============================
# Query record + bounded pool
# ==============================

@dataclass
class QueryRecord:
    """Lightweight record kept on the driver for queue items."""
    raw_prompt_ids: np.ndarray          # (seq_len,) token ids (preferred) or plain text
    # cached tensors for fast batching (text-only)
    input_ids: Optional[torch.Tensor]
    attention_mask: Optional[torch.Tensor]
    position_ids: Optional[torch.Tensor]
    # teacher-provided ground truth (task-dependent)
    gt: Optional[object] = None
    reward: float = 0.5                 # uniform init
    # policy-estimated reward for augmented queries
    est_reward: Optional[float] = None
    meta: dict = None                   # arbitrary metadata


class ThreadSafeQueryPool:
    """
    Step 1 + Steps 6/2 support: a simple multi-producer/consumer sampler
    WITH a hard max size and simple eviction/admission control.
    """
    # TODO: We need to reimplement this class to support: (1) heap-based query pool management; and (2) reward-based batch sampling.
    # For the first one, we need to build two min heaps as the base structure of the pool: the first one for the queries with the lowest rewards, and the second one for the queries with the highest rewards.
    # When we insert a new query, by default we insert it into the second heap, and if the size of the second heap exceeds the max size, we pop the lowest reward query from the first heap and insert it into the second heap.
    # This can be regarded as a priority queue, where the priority is determined by the reward of the query.
    # Then the second heap maintains a pool of queries that has higher rewards (which means the difficulty is lower), and the first heap maintains a pool of queries that has lower rewards (which means the difficulty is higher).
    # For the first heap, the top node is the query with the lowest reward, and for the second heap, the top node is with medium reward, as its reward is lower than bottom nodes.
    # Thus, for batch sampling, we can sample from the close-top levels from the second heap and the close-bottom levels from the first heap, this ensures that we are sampling with medium difficulty queries, and the sampled queries are diverse enough.
    # To enable randomness, if we are taking n queries, we do take out 2n queries from both heaps, and then randomly sample n queries from the 2n queries.
    # Then we re-insert the unsampled queries back to the heaps, this also ensures the randomness of the sampled queries and the diversity of queries in the pool.

    def __init__(self, max_size: int = 30000, eviction_policy: str = "reject_new"):
        """
        eviction_policy: "reject_new" | "drop_oldest" | "drop_random"
        """
        self._lock = threading.Lock()
        self._items: List[QueryRecord] = []
        self._max_size = max_size
        self._eviction_policy = eviction_policy

    def set_max_size(self, max_size: int):
        with self._lock:
            self._max_size = max_size

    def capacity_remaining(self) -> int:
        with self._lock:
            return max(0, self._max_size - len(self._items))

    def initialize_uniform(self, seed_items: List[QueryRecord], uniform_reward: float = 0.5):
        with self._lock:
            for it in seed_items:
                it.reward = uniform_reward
            # If seed is larger than cap, trim by policy
            if len(seed_items) > self._max_size:
                if self._eviction_policy == "reject_new":
                    self._items = seed_items[: self._max_size]
                elif self._eviction_policy == "drop_oldest":
                    # keep the most recent (tail)
                    self._items = seed_items[-self._max_size:]
                else:  # drop_random
                    idx = np.random.choice(
                        len(seed_items), size=self._max_size, replace=False)
                    self._items = [seed_items[i] for i in idx]
            else:
                self._items = list(seed_items)

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def _evict_if_needed(self, incoming_count: int) -> bool:
        """Return True if we can accept 'incoming_count' after evicting per policy; False to reject."""
        with self._lock:
            need = incoming_count - max(0, self._max_size - len(self._items))
            if need <= 0:
                return True
            if self._eviction_policy == "reject_new":
                return False
            if self._eviction_policy == "drop_oldest":
                del self._items[: min(need, len(self._items))]
                return len(self._items) + incoming_count <= self._max_size
            if self._eviction_policy == "drop_random":
                for _ in range(min(need, len(self._items))):
                    j = np.random.randrange(len(self._items))
                    self._items.pop(j)
                return len(self._items) + incoming_count <= self._max_size
            return False

    def add_many(self, items: List[QueryRecord]):
        # Try to evict (or reject) to make room
        if not items:
            return
        if not self._evict_if_needed(len(items)):
            # reject new when full
            return
        with self._lock:
            to_fit = min(len(items), self._max_size - len(self._items))
            if to_fit > 0:
                self._items.extend(items[:to_fit])

    def sample_batch(self, k: int) -> List[QueryRecord]:
        # TODO: reward based sampling
        with self._lock:
            n = len(self._items)
            if n == 0:
                return []
            if k >= n:
                return [self._copy(it) for it in self._items]
            idx = np.random.choice(n, size=k, replace=False)
            return [self._copy(self._items[i]) for i in idx]

    @staticmethod
    def _copy(it: QueryRecord) -> QueryRecord:
        return QueryRecord(
            raw_prompt_ids=it.raw_prompt_ids,
            input_ids=it.input_ids,
            attention_mask=it.attention_mask,
            position_ids=it.position_ids,
            gt=it.gt,
            reward=it.reward,
            est_reward=it.est_reward,
            meta=dict(it.meta or {}),
        )


# ==============================
# Asynchronous GPT annotator
# ==============================

class AsyncTeacherAnnotator(threading.Thread):
    """
    Steps 5 & 6: background thread that sends augmented queries to OpenAI GPT,
    receives (query, ground-truth, est_reward), and adds them into the pool.
    """

    def __init__(self, trainer_ref: "RayDAPOTrainer", poll_interval: float = 0.1):
        super().__init__(daemon=True)
        self.trainer_ref = trainer_ref
        self.queue = deque()          # holds DataProto from policy augmentation
        self.queue_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.poll_interval = poll_interval

        # OpenAI client (key from env first, fallback to key module)
        api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
        if not api_key:
            print(
                "[AsyncTeacherAnnotator] WARNING: OPENAI_API_KEY not set; teacher calls will fail.")
        self.client = OpenAI(api_key=api_key)

    def enqueue_aug(self, aug_proto: "DataProto"):
        with self.queue_lock:
            self.queue.append(aug_proto)

    def shutdown(self):
        self.stop_event.set()

    def _make_prompt(self, question: str) -> str:
        """
        Prompt template: ask GPT to compute the answer to a math problem.
        """
        return (
            "You are a precise math solver.\n"
            f"Problem: {question}\n\n"
            "Return ONLY the final numeric answer with no explanation or additional text."
        )

    def _extract_text(self, resp) -> str:
        # Try Responses API structure
        try:
            return resp.output[0].content[0].text.strip()
        except Exception:
            pass
        # Try Chat Completions-like structure
        try:
            return resp.choices[0].message["content"].strip()
        except Exception:
            pass
        # Fallback
        return str(resp).strip()

    def run(self):
        while not self.stop_event.is_set():
            aug_proto = None
            with self.queue_lock:
                if self.queue:
                    aug_proto = self.queue.popleft()
            if aug_proto is None:
                self.stop_event.wait(self.poll_interval)
                continue

            # === Teacher annotation via OpenAI API ===
            try:
                # TODO: we need to pass the original reward as input for add_many function.
                # If reward is none or dummy (not calculated by policy model), we skip that query and don't do augmentation yet.
                raw_prompts = aug_proto.non_tensor_batch["raw_prompt_ids"]
                teacher_answers = []
                for q in raw_prompts:
                    prompt = self._make_prompt(q)
                    response = self.client.responses.create(
                        model="gpt-5-mini",
                        input=prompt
                    )
                    answer_text = self._extract_text(response)
                    teacher_answers.append(answer_text)

                # Attach teacher answers into DataProto
                aug_proto.non_tensor_batch["teacher/gt"] = teacher_answers

                # If pool is at capacity, back off to avoid busy loop
                if self.trainer_ref.query_pool.capacity_remaining() <= 0:
                    self.stop_event.wait(self.poll_interval)
                    continue

            except Exception as e:
                print(f"[AsyncTeacherAnnotator] OpenAI API error: {e}")
                continue

            # Convert annotated DataProto -> QueryRecord list and add to pool
            try:
                new_items: List["QueryRecord"] = self.trainer_ref._proto_to_query_records(
                    aug_proto)
                self.trainer_ref.query_pool.add_many(new_items)
            except Exception as e:
                print(f"[AsyncTeacherAnnotator] convert/add error: {e}")


# ==============================
# Trainer with throttled augmentation (text-only)
# ==============================

class RayDAPOTrainer(RayPPOTrainer):
    """
    PPO + dynamic queue/augmentation (text-only).
    Includes policy-driven `generate_augmented_queries` that:
      - decodes original problems,
      - builds your augmentation prompt,
      - calls the actor to generate a NEW problem,
      - parses the '#New Problem#' section,
      - tokenizes NEW problems into a DataProto for the async teacher.
    """

    def __init__(
        self,
        *,
        config,
        tokenizer,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
        **kwargs,   # ← keep this to be future-proof
    ):
        # if you need them in augmentation helpers:
        self.tokenizer = tokenizer
        self.processor = processor

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
            **kwargs,
        )
    # ----------------------------
    # Helpers: DataProto <-> QueryRecord
    # ----------------------------

    def _seed_records_from_loader(self) -> List[QueryRecord]:
        seed: List[QueryRecord] = []
        seed_cap = getattr(self.config.dynamic_data, "seed_cap", 0) or 0
        cap_left = seed_cap if seed_cap > 0 else float("inf")
        for batch_dict in self.train_dataloader:
            if cap_left <= 0:
                break
            dp = DataProto.from_single_dict(batch_dict)
            for i in range(len(dp.batch["input_ids"])):
                seed.append(
                    QueryRecord(
                        raw_prompt_ids=dp.non_tensor_batch["raw_prompt_ids"][i],
                        input_ids=dp.batch["input_ids"][i],
                        attention_mask=dp.batch["attention_mask"][i],
                        position_ids=dp.batch.get("position_ids", None)[
                            i] if "position_ids" in dp.batch else None,
                        gt=dp.non_tensor_batch.get(
                            "gt", [None]*len(dp.batch["input_ids"]))[i] if "gt" in dp.non_tensor_batch else None,
                        reward=0.5,
                        est_reward=None,
                        meta={"source": "seed"},
                    )
                )
                cap_left -= 1
                if cap_left <= 0:
                    break
        return seed

    def _records_to_dataproto(self, recs: List[QueryRecord]) -> DataProto:
        input_ids = torch.stack([r.input_ids for r in recs], dim=0)
        attention_mask = torch.stack([r.attention_mask for r in recs], dim=0)
        position_ids = torch.stack(
            [r.position_ids for r in recs], dim=0) if recs[0].position_ids is not None else None
        nt = {
            "raw_prompt_ids": np.array([r.raw_prompt_ids for r in recs], dtype=object),
            "driver_reward": np.array([r.reward for r in recs], dtype=np.float32),
            "driver_est_reward": np.array(
                [r.est_reward if r.est_reward is not None else np.nan for r in recs],
                dtype=np.float32
            ),
            "driver_gt": np.array([r.gt for r in recs], dtype=object),
        }
        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if position_ids is not None:
            batch["position_ids"] = position_ids
        return DataProto(batch=batch, non_tensor_batch=nt, meta_info={})

    def _proto_to_query_records(self, dp: DataProto) -> List[QueryRecord]:
        """
        Convert teacher-annotated augmented samples into QueryRecords (Steps 5–6).
        NOTE: For robustness, we derive raw_prompt_ids from the token ids (batch["input_ids"]),
        not from dp.non_tensor_batch["raw_prompt_ids"] (which may be plain text in some paths).
        """
        recs: List[QueryRecord] = []
        gt_list = dp.non_tensor_batch.get(
            "teacher/gt", [None] * len(dp.batch["input_ids"]))
        est_list = dp.non_tensor_batch.get(
            "policy/est_reward", [None] * len(dp.batch["input_ids"]))
        for i in range(len(dp.batch["input_ids"])):
            token_ids_np = dp.batch["input_ids"][i].detach().cpu().numpy()
            recs.append(
                QueryRecord(
                    raw_prompt_ids=token_ids_np,  # ensure tokens, not text
                    input_ids=dp.batch["input_ids"][i],
                    attention_mask=dp.batch["attention_mask"][i],
                    position_ids=dp.batch.get("position_ids", None)[
                        i] if "position_ids" in dp.batch else None,
                    gt=gt_list[i],
                    reward=0.5,
                    est_reward=float(est_list[i]) if est_list[i] is not None and not (
                        isinstance(est_list[i], float) and np.isnan(
                            est_list[i])
                    ) else None,
                    meta={"source": "augment"},
                )
            )
        return recs

    # ----------------------------
    # Helpers: text encode/decode
    # ----------------------------
    def _decode_tokens_to_text(self, token_seq: np.ndarray) -> str:
        # token_seq may be np.ndarray of ints
        try:
            return self.tokenizer.decode(list(token_seq), skip_special_tokens=True)
        except Exception:
            # fallback if already text
            return str(token_seq)

    def _tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        # Respect a prompt length cap if available
        max_len = int(getattr(self.config.data, "max_prompt_length", 2048))
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    # ----------------------------
    # policy-driven augmentation
    # ----------------------------
    def generate_augmented_queries(
        self,
        source_batch: DataProto,
        num_per_prompt: int = 1,
        aug_cfg: Optional[Dict] = None
    ) -> DataProto:
        """
        Use the current policy model to create NEW math problems from originals.
        Steps:
          1) Decode each original problem to text.
          2) Build the user's augmentation prompt with '#Original Problem#' ... '#New Problem#'.
          3) Tokenize and call actor.generate_sequences() to get completions.
          4) Extract only the NEW problem text (after '#New Problem#').
          5) Tokenize NEW problems and return as DataProto ready for teacher annotation.

        Returns:
          DataProto with batch = tokenized NEW problems (input_ids, attention_mask)
                    non_tensor_batch["raw_prompt_ids"] = NEW problem TEXT (for teacher),
                    non_tensor_batch["policy/est_reward"] = list of None/estimates (optional).
        """
        if num_per_prompt <= 0:
            # Return empty batch
            return DataProto(
                batch={"input_ids": torch.empty(
                    0, dtype=torch.long), "attention_mask": torch.empty(0, dtype=torch.long)},
                non_tensor_batch={
                    "raw_prompt_ids": np.array([], dtype=object)},
                meta_info={}
            )

        # 1) Gather unique originals (dedup repeated rollouts)
        raw_ids_list = list(source_batch.non_tensor_batch["raw_prompt_ids"])
        seen = set()
        originals: List[np.ndarray] = []
        for arr in raw_ids_list:
            key = tuple(np.asarray(arr).tolist())
            if key not in seen:
                seen.add(key)
                originals.append(np.asarray(arr))

        # 2) Build augmentation prompts from user's template
        TEMPLATE = (
            "Given a math problem and its proposed answer, your task is to generate a new question that is similar in style and difficulty to the original problem, but distinct and not a direct copy.\n"
            "The new question should challenge the problem-solving skills of someone familiar with the original problem, yet be solvable using similar logic or mathematical concepts.\n"
            "Generate the problem directly after #New Problem#, maintaining a similar length and complexity, but with slightly more intricate reasoning steps or an additional constraint (e.g., an extra condition, range restriction, or parameter to consider).\n"
            "Do not include any explanations or answers, only the new problem statement.\n"
            "You may alter numerical values, change the context, or introduce minor structural variations, but ensure the core mathematical concepts remain comparable.\n\n"
            "#Original Problem#\n{original_problem}\n\n#New Problem#"
        )
        # justify whether the augmented query is more complex or easier
        # assume the current difficulty is 1, estimate the new query's difficulty
        # as 1.2 (20% more complex) or 0.8 (20% easier), if 1.2, we divide the reward by 1.2
        # if difficulty is not stable, just try query rewriting

        aug_prompts: List[str] = []
        for ids in originals:
            original_text = self._decode_tokens_to_text(ids)
            for _ in range(num_per_prompt):
                aug_prompts.append(TEMPLATE.format(
                    original_problem=original_text))

        # 3) Tokenize augmentation prompts and call policy to generate NEW problems
        enc = self._tokenize_texts(aug_prompts)

        # Build pop keys safely (position_ids may be absent downstream)
        aug_gen_batch = DataProto(
            batch={"input_ids": enc["input_ids"],
                   "attention_mask": enc["attention_mask"]},
            non_tensor_batch={
                "raw_aug_prompt_text": np.array(aug_prompts, dtype=object),
                # for slicing generated part
                "aug_prompt_len": enc["attention_mask"].sum(dim=1).cpu().numpy(),
            },
            meta_info={}
        )
        gen_out = self.actor_rollout_wg.generate_sequences(aug_gen_batch)

        # 4) Extract generated NEW problem text
        if "sequences" in gen_out.batch:
            seq_tensor = gen_out.batch["sequences"]
        else:
            seq_tensor = gen_out.batch.get("input_ids", None)
        if seq_tensor is None:
            raise RuntimeError(
                "generate_sequences did not return 'sequences' or 'input_ids'.")

        prompt_lens = aug_gen_batch.non_tensor_batch["aug_prompt_len"]
        new_texts: List[str] = []
        for i in range(seq_tensor.size(0)):
            seq_ids = seq_tensor[i].detach().cpu().tolist()
            p_len = int(prompt_lens[i]) if i < len(prompt_lens) else 0
            gen_ids = seq_ids[p_len:] if p_len < len(seq_ids) else []
            text = self.tokenizer.decode(
                gen_ids, skip_special_tokens=True).strip()
            # Parse after '#New Problem#'
            marker = "#New Problem#"
            if marker in text:
                text = text.split(marker, 1)[1].strip()
            new_texts.append(text.strip())

        # 5) Tokenize NEW problems (these become the *queries* we send to teacher)
        new_enc = self._tokenize_texts(new_texts)
        nt = {
            # text for teacher prompt
            "raw_prompt_ids": np.array(new_texts, dtype=object),
            "policy/est_reward": np.array([None] * len(new_texts), dtype=object),
        }
        aug_proto_for_teacher = DataProto(
            batch={"input_ids": new_enc["input_ids"],
                   "attention_mask": new_enc["attention_mask"]},
            non_tensor_batch=nt,
            meta_info={}
        )
        return aug_proto_for_teacher

    # ----------------------------
    # Main training loop
    # ----------------------------
    def fit(self):
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        self._load_checkpoint()

        # dynamic data plumbing
        dyn_en = bool(getattr(self.config, "dynamic_data",
                      {}).get("enable", False))

        max_pool_size = int(
            getattr(self.config.dynamic_data, "max_pool_size", 30000))
        eviction_policy = str(
            getattr(self.config.dynamic_data, "eviction_policy", "reject_new"))
        self.query_pool = ThreadSafeQueryPool(
            max_size=max_pool_size, eviction_policy=eviction_policy)

        self.teacher_annotator: Optional[AsyncTeacherAnnotator] = None

        if dyn_en:
            seed_uniform = float(
                getattr(self.config.dynamic_data, "uniform_reward", 0.5))
            seed_records = self._seed_records_from_loader()
            self.query_pool.initialize_uniform(
                seed_records, uniform_reward=seed_uniform)

            if not hasattr(self, "teacher_wg"):
                self.teacher_wg = None

            self.teacher_annotator = AsyncTeacherAnnotator(self)
            self.teacher_annotator.start()

        # pre-training validation
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                if self.teacher_annotator is not None:
                    self.teacher_annotator.shutdown()
                    self.teacher_annotator.join(timeout=2.0)
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps,
                            initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        aug_cfg = getattr(self.config, "augmentation", {})
        do_augment = dyn_en and bool(aug_cfg.get("enable", True))

        for epoch in range(self.config.trainer.total_epochs):
            steps_per_epoch = getattr(
                self.config.trainer, "steps_per_epoch", None)
            if not dyn_en:
                data_iterable = self.train_dataloader
            else:
                data_iterable = range(
                    steps_per_epoch if steps_per_epoch is not None else 10**9)

            for batch_dict in data_iterable:
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # Step 2: sample batch from queue (text-only)
                if dyn_en:
                    want = self.config.data.train_batch_size
                    sampled = self.query_pool.sample_batch(k=want)
                    if not sampled:
                        tqdm.write("[dynamic] queue is empty; waiting…")
                        if torch.cuda.is_available():
                            torch.cuda._sleep(int(1e6))
                        else:
                            time.sleep(0.001)
                        continue
                    new_batch = self._records_to_dataproto(sampled)
                else:
                    new_batch: DataProto = DataProto.from_single_dict(
                        batch_dict)

                num_gen_batches += 1

                # pop keys for generation (text-only), guarding position_ids
                pop_keys = ["input_ids", "attention_mask"]
                if "position_ids" in new_batch.batch:
                    pop_keys.append("position_ids")
                gen_batch = new_batch.pop(
                    batch_keys=pop_keys,
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )
                gen_batch = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

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

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(
                                dim=-1)
                            new_batch.pop(batch_keys=list(
                                gen_baseline_output.batch.keys()))
                            new_batch.batch["reward_baselines"] = reward_baseline_tensor
                            del gen_baseline_batch, gen_baseline_output

                    # Correct batch size when generating uids
                    bsz = int(new_batch.batch["input_ids"].size(0))
                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(bsz)], dtype=object
                    )

                    new_batch = new_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    # Step 3 (cont.): reward calc
                    with marked_timer("reward", timing_raw, "yellow"):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(
                                new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        reward_extra_infos_dict: dict
                        try:
                            reward_result = self.reward_fn(
                                new_batch, return_dict=True)
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get(
                                "reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v)
                                 for k, v in reward_extra_infos_dict.items()}
                            )

                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    # Step 4: policy-driven augmentation (THROTTLED)
                    if do_augment:
                        try:
                            self.query_pool.set_max_size(
                                int(getattr(self.config.dynamic_data, "max_pool_size", 30000)))
                            remain = self.query_pool.capacity_remaining()
                            if remain <= 0:
                                effective_num_per_prompt = 0
                            else:
                                want_per_prompt = int(
                                    aug_cfg.get("num_per_prompt", 2))
                                bsz_curr = len(new_batch.batch["input_ids"])
                                max_per_prompt_by_capacity = max(
                                    0, remain // max(1, bsz_curr))
                                effective_num_per_prompt = max(
                                    0, min(want_per_prompt, max_per_prompt_by_capacity))

                            if effective_num_per_prompt > 0:
                                aug_proto = self.generate_augmented_queries(
                                    source_batch=new_batch,
                                    num_per_prompt=effective_num_per_prompt,
                                    aug_cfg=aug_cfg,
                                )
                                if self.teacher_annotator is not None:
                                    if self.query_pool.capacity_remaining() > 0:
                                        self.teacher_annotator.enqueue_aug(
                                            aug_proto)
                            # else skip augmentation

                        except Exception as e:
                            print(f"[dynamic] augmentation error: {e}")

                    # (DAPO group filtering) unchanged
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
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {uid: np.std(
                            vals) for uid, vals in prompt_uid2metric_vals.items()}

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
                        batch = new_batch if batch is None else DataProto.concat(
                            [batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                is_last_step = self.gen_steps >= self.total_training_steps
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

                    # Step 7: PPO updating (unchanged)
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(
                            batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(
                            loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
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
                            critic_output = self.critic_wg.update_critic(batch)
                        metrics.update(reduce_metrics(
                            critic_output.meta_info["metrics"]))

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(
                                batch)
                        metrics.update(reduce_metrics(
                            actor_output.meta_info["metrics"]))

                # validate / save (unchanged)
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # metrics
                metrics.update(compute_data_metrics(
                    batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(
                    batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(
                    batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)

                metrics["train/num_gen_batches"] = num_gen_batches
                metrics["dynamic/pool_size"] = self.query_pool.size() if dyn_en else 0
                metrics["dynamic/capacity_remaining"] = self.query_pool.capacity_remaining() if dyn_en else 0
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    if self.teacher_annotator is not None:
                        self.teacher_annotator.shutdown()
                        self.teacher_annotator.join(timeout=2.0)
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

        checkpoint_dir = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)

        if self.teacher_annotator is not None:
            self.teacher_annotator.shutdown()
            self.teacher_annotator.join(timeout=2.0)
