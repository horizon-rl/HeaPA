#!/usr/bin/env python3
"""
slime/train.py — HeaPA-aware entry point for slime training.

Pre-parses --heapa-* arguments (and any other args not recognized by the
slime/Megatron backend) before slime's arg parser sees sys.argv, then
injects them back into the parsed namespace so that HeaPADataSource can
read them via getattr(args, ...).

Usage (via run_heapa.sh):
    python <repo>/slime/train.py --heapa-pool-max-size 1000000 ... <slime args>
"""
import argparse
import importlib.util
import os
import sys

# ── 1. Pre-parse and strip custom args before slime's parser runs ─────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--heapa-pool-max-size",    type=int,   default=30000,       dest="heapa_pool_max_size")
_parser.add_argument("--heapa-low-fraction",     type=float, default=0.5,         dest="heapa_low_fraction")
_parser.add_argument("--heapa-mixed-sampling",   action="store_true",             dest="heapa_mixed_sampling")
_parser.add_argument("--heapa-teacher-enabled",  action="store_true",             dest="heapa_teacher_enabled")
_parser.add_argument("--heapa-teacher-model",    type=str,   default="gpt-4o-mini", dest="heapa_teacher_model")
_parser.add_argument("--heapa-teacher-workers",  type=int,   default=4,           dest="heapa_teacher_workers")
_parser.add_argument("--heapa-teacher-hard-lo",  type=float, default=0.1,         dest="heapa_teacher_hard_lo")
_parser.add_argument("--heapa-teacher-hard-hi",  type=float, default=0.7,         dest="heapa_teacher_hard_hi")
_parser.add_argument("--heapa-reseed-threshold", type=int,   default=100,         dest="heapa_reseed_threshold")
# --loss-agg-mode is not supported by the Megatron backend in this slime version;
# consume it here to avoid "unrecognized argument" errors.
_parser.add_argument("--loss-agg-mode",          type=str,   default="token-mean", dest="loss_agg_mode")

_custom_ns, _remaining = _parser.parse_known_args()
sys.argv = [sys.argv[0]] + _remaining  # slime/Megatron only sees the remaining args

# ── 2. Patch slime's read_file to inject a 'label' field for verl-format data ─
# Datasets produced by the verl pipeline store the ground truth in
# reward_model['ground_truth'] rather than a top-level 'label' column.
# Patching read_file ensures every Dataset call (eval + training) gets a usable
# 'label' string without modifying the parquet files.
import slime.utils.data as _slime_data

_orig_read_file = _slime_data.read_file


def _patched_read_file(path):
    for row in _orig_read_file(path):
        if not row.get("label"):
            rm = row.get("reward_model")
            if isinstance(rm, dict) and rm.get("ground_truth"):
                row = dict(row)
                row["label"] = str(rm["ground_truth"])
        yield row


_slime_data.read_file = _patched_read_file

# ── 3. Patch slime's parse_args to inject custom attrs into the result ─────────
import slime.utils.arguments as _slime_arguments

_orig_parse_args = _slime_arguments.parse_args


def _parse_args_with_heapa(*args, **kwargs):
    ns = _orig_parse_args(*args, **kwargs)
    for key, val in vars(_custom_ns).items():
        setattr(ns, key, val)
    return ns


_slime_arguments.parse_args = _parse_args_with_heapa

# ── 3. Delegate to slime's original train.py entry point ─────────────────────
_slime_root = os.environ.get("SLIME_ROOT", "/root/slime")
_train_py = os.path.join(_slime_root, "train.py")

spec = importlib.util.spec_from_file_location("__main__", _train_py)
_mod = importlib.util.module_from_spec(spec)
_mod.__name__ = "__main__"
spec.loader.exec_module(_mod)
