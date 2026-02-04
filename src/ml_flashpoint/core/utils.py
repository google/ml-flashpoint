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

import logging
import os
import time
from contextlib import contextmanager
from typing import TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


def get_env_var_prefix() -> str:
    """Returns the prefix for this application's environment variables.

    Returns:
        The env var prefix string.
    """
    return "MLFLASHPOINT"


def get_accelerator_count() -> int:
    """Returns the number of accelerators available in the host.

    Returns:
        int: The number of accelerators.

    Raises:
        RuntimeError: If no accelerators are available.
    """
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    else:
        raise RuntimeError("Cannot determine number of accelerators for CPU-only training.")


def get_num_of_nodes() -> int:
    """
    Calculates the number of nodes in a distributed job without using LOCAL_WORLD_SIZE.

    The function follows this priority order:
    1. Reads the `NNODES` environment variable.
    2. Reads the `SLURM_NNODES` environment variable.
    3. Calculates from `world_size` / `torch.cuda.device_count()`,
       ASSUMING a one-process-per-GPU setup.

    Returns:
        int: The total number of nodes.
    """
    # Priority 1 & 2 remain the same
    if "NNODES" in os.environ:
        return int(os.environ["NNODES"])

    if "SLURM_NNODES" in os.environ:
        return int(os.environ["SLURM_NNODES"])

    # Priority 3: Fallback calculation
    if not dist.is_available() or not dist.is_initialized():
        return 1

    world_size = dist.get_world_size()

    nprocs_per_node = get_accelerator_count()
    if nprocs_per_node == 0:
        raise RuntimeError("torch.cuda.device_count() returned 0.")

    if world_size % nprocs_per_node != 0:
        raise RuntimeError(
            f"Inconsistent setup: world_size ({world_size}) is not "
            f"evenly divisible by the GPU count per node ({nprocs_per_node})."
        )

    return world_size // nprocs_per_node


def get_env_val_bool(env_var_name: str, default_val: bool) -> bool:
    """Returns the environment variable value for the given env_prop_name, prefixed with this app's custom prefix.

    Expects the env var value to be "true" or "false" (case-insensitive).

    Uses the provided default_val when the environment variable is not found.

    Args:
        env_var_name: The suffix of the environment variable name.
        default_val: The default value to return if the environment variable is missing.

    Returns:
        The boolean value of the environment variable or the default value.
    """
    return str(os.environ.get(f"{get_env_var_prefix()}_{env_var_name}", default_val)).lower() == "true"


def get_env_val_str(env_var_name: str, default_val: str) -> str:
    """Returns the environment variable value for the given env_prop_name, prefixed with this app's custom prefix.

    Uses the provided default_val when the environment variable is not found.

    Args:
        env_var_name: The suffix of the environment variable name.
        default_val: The default value to return if the environment variable is missing.

    Returns:
        The string value of the environment variable or the default value.
    """
    return os.environ.get(f"{get_env_var_prefix()}_{env_var_name}", default_val)


def get_env_val_int(env_var_name: str, default_val: int) -> int:
    """Returns the environment variable value for the given env_prop_name, prefixed with this app's custom prefix.

    Expects the env var value to be an integer.

    Uses the provided default_val when the environment variable is not found or cannot be converted to an integer.

    Args:
        env_var_name: The suffix of the environment variable name.
        default_val: The default value to return if the environment variable is missing or invalid.

    Returns:
        The integer value of the environment variable or the default value.
    """
    env_val = os.environ.get(f"{get_env_var_prefix()}_{env_var_name}", default_val)
    if env_val is None:
        return default_val
    try:
        return int(env_val)
    except ValueError:
        return default_val


@contextmanager
def log_execution_time(logger: logging.Logger, name: str, level: int = logging.DEBUG):
    """Simple context manager for timing functions/code blocks.

    Args:
        logger: The logger to use for recording the time.
        name: The name of the operation being timed.
        level: The logging level to use. Defaults to logging.DEBUG.

    Yields:
        None.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.log(level, "%s took %.4fs", name, time.perf_counter() - start)
