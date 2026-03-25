import os
import socket
import logging

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available

# Import the improved trainer
from .dapo_ray_trainer_augment_heap_reward_propagation import RayDAPOTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    run_dapo_improved(config)


def run_dapo_improved(config) -> None:
    if not ray.is_initialized():
        # Environment variables for optimal performance
        env_vars = {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
            "VLLM_LOGGING_LEVEL": "WARN",
            "OMP_NUM_THREADS": "1",  # Prevent thread oversubscription
            "MKL_NUM_THREADS": "1",
        }

        # Add OpenAI API key if available (for teacher annotation)
        if os.environ.get("OPENAI_API_KEY"):
            env_vars["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        elif config.get("dynamic_data", {}).get("enable", False):
            # Try to get from key module if dynamic data is enabled
            try:
                from key import OPENAI_API_KEY
                if OPENAI_API_KEY:
                    env_vars["OPENAI_API_KEY"] = OPENAI_API_KEY
                    logger.info("Loaded OPENAI_API_KEY from key module")
            except ImportError:
                logger.warning(
                    "OPENAI_API_KEY not found - teacher annotation will fail if enabled")

        ray.init(
            runtime_env={"env_vars": env_vars},
            num_cpus=config.ray_init.get("num_cpus", None),
            num_gpus=config.ray_init.get("num_gpus", None),
            object_store_memory=config.ray_init.get(
                "object_store_memory", None),
        )

    if (
        is_cuda_available
        and OmegaConf.select(config.trainer, "profile_steps") is not None
        and len(OmegaConf.select(config.trainer, "profile_steps")) > 0
    ):
        nsight_options = OmegaConf.to_container(
            config.trainer.controller_nsight_options)
        runner = TaskRunner.options(
            runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()

    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local
        from transformers import AutoTokenizer, AutoProcessor

        print(
            f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        model_path = OmegaConf.select(config, "actor_rollout_ref.model.path")
        if not model_path:
            raise ValueError("config.actor_rollout_ref.model.path is missing")

        if os.path.isdir(model_path):
            local_path = os.path.abspath(model_path)
            tokenizer = AutoTokenizer.from_pretrained(
                local_path, trust_remote_code=True, local_files_only=True
            )
            try:
                processor = AutoProcessor.from_pretrained(
                    local_path, trust_remote_code=True, local_files_only=True, use_fast=True
                )
            except Exception:
                processor = None
        else:
            local_path = copy_to_local(model_path)
            from verl.utils import hf_processor, hf_tokenizer
            tokenizer = hf_tokenizer(local_path)
            processor = hf_processor(local_path, use_fast=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")

        from verl.single_controller.ray import RayWorkerGroup

        # define worker classes
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            assert config.critic.strategy in {"fsdp", "fsdp2"}

            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

            ray_worker_group_cls = RayWorkerGroup

        else:
            raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # we should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data
        if config.reward_model.enable:
            if config.reward_model.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward_model.strategy == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(
                RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(
                ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        reward_fn = load_reward_manager(
            config,
            tokenizer,
            0,
            max_resp_len=config.data.max_response_length,
            overlong_buffer_cfg=config.reward_model.get("overlong_buffer", {}),
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = None
        if config.trainer.get("test_freq", 0) > 0:
            val_reward_fn = load_reward_manager(
                config,
                tokenizer,
                1,
                max_resp_len=config.data.max_response_length,
                overlong_buffer_cfg=config.reward_model.get(
                    "overlong_buffer", {}),
            )

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping)

        # Log dynamic data configuration status
        if config.get("dynamic_data", {}).get("enable", False):
            logger.info("Dynamic data augmentation is ENABLED")
            logger.info(
                f"  Teacher model: {config.dynamic_data.get('teacher_model', 'gpt-3.5-turbo')}")
            logger.info(
                f"  Max pool size: {config.dynamic_data.get('max_pool_size', 30000)}")
            logger.info(
                f"  Augmentation enabled: {config.get('augmentation', {}).get('enable', False)}")
        else:
            logger.info(
                "Dynamic data augmentation is DISABLED - using standard PPO training")

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )

        try:
            trainer.init_workers()
            trainer.fit()
        finally:
            # Ensure proper cleanup
            if hasattr(trainer, '_cleanup_teacher_annotator'):
                trainer._cleanup_teacher_annotator()
            if hasattr(trainer, 'augmentation_logger'):
                if trainer.augmentation_logger:
                    trainer.augmentation_logger.flush_all()
                    logger.info("Augmentation logs saved successfully")


if __name__ == "__main__":
    main()
