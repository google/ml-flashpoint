# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom ML Flashpoint logging configuration."""

import logging
import multiprocessing
import sys

import torch
from typing_extensions import override

from ml_flashpoint.core import utils

_MISSING_NONNEG_NUMERIC_VAL = -1
"""A sentinel integer value for missing numerical values (that are expected to be non-negative when present)."""

# Private static state to track the current step solely for logging purposes.
# -1 is the default sentinel (invalid) value.
_TRAINING_STEP = multiprocessing.Value("i", _MISSING_NONNEG_NUMERIC_VAL)


def update_training_step(new_val: int):
    """Updates the global training step value used in logs.

    Args:
        new_val: The new training step value. Any negative value is considered
            invalid and will be logged as "N/A".
    """
    with _TRAINING_STEP.get_lock():
        _TRAINING_STEP.value = new_val


class TrainingContextFormatter(logging.Formatter):
    """A logging formatter that includes useful contextual information in the log records."""

    @override
    def format(self, record):
        """Formats the log record to include the rank and current training step.

        Args:
            record: The log record to format.

        Returns:
            The formatted log record as a string.
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else _MISSING_NONNEG_NUMERIC_VAL
        record.rank = rank
        step_val = _TRAINING_STEP.value
        record.curr_step = step_val
        return super().format(record)


def get_logger(name: str, stream=sys.stderr) -> logging.Logger:
    """Get a logger with a custom format that includes the rank.

    Args:
        name: The name of the logger.
        stream: The stream to write log records to. Defaults to sys.stderr.

    Returns:
        A logger with a custom format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=stream)
        formatter = TrainingContextFormatter(
            "[MLF %(asctime)s %(levelname)s Step=%(curr_step)s Rank=%(rank)s %(name)s:%(lineno)d] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        log_level_str = utils.get_env_val_str("LOG_LEVEL", "DEBUG")
        log_level = logging._nameToLevel.get(log_level_str.upper(), logging.DEBUG)
        logger.setLevel(log_level)
    logger.propagate = False
    return logger
