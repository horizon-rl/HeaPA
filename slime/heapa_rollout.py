"""
heapa_rollout.py - Custom rollout function for HeaPA slime training.

Usage (slime CLI):
  --rollout-function-path slime.heapa_rollout.generate_rollout

This wraps slime's built-in sglang rollout but hooks into the HeaPADataSource
to update pool rewards after each rollout and trigger teacher augmentation.
"""
from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.sglang_rollout import eval_rollout, generate_rollout_async
from slime.utils.async_utils import run

logger = logging.getLogger(__name__)


def generate_rollout(
    args: Namespace,
    rollout_id: int,
    data_source: Any,
    evaluation: bool = False,
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """
    HeaPA rollout function.

    Differences from the default sglang rollout:
      - After generation, calls data_source.process_scored_samples() to update the
        heap-based pool with actual rewards and trigger teacher augmentation.
      - data_source is expected to be a HeaPADataSource instance.
    """
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    # Use slime's built-in async generation; pass data_source.get_samples as the callable
    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))

    # HeaPA: update pool with scored samples
    if hasattr(data_source, "process_scored_samples"):
        try:
            data_source.process_scored_samples(output.samples)
        except Exception as e:
            logger.warning(f"[HeaPA] process_scored_samples failed: {e}", exc_info=True)

    # Re-queue aborted/partial rollout samples
    data_source.add_samples(aborted_samples)

    return output
