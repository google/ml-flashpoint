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

"""Configuration for the Megatron Bridge ML Flashpoint adapter.

Megatron Bridge instantiates a custom checkpoint manager through
:func:`megatron.bridge.training.checkpointing.create_checkpoint_manager`, which
only passes the framework's own ``CheckpointConfig``. There is therefore no
place in the Bridge config to carry ML Flashpoint's own knobs, so they are
resolved from (in order of precedence):

1. an explicit :func:`configure` call made before training starts, and
2. ``MLFLASHPOINT_*`` environment variables.
"""

import dataclasses
from typing import Optional

from ml_flashpoint.core import utils
from ml_flashpoint.core.checkpoint_saver import DEFAULT_INITIAL_BUFFER_SIZE_BYTES
from ml_flashpoint.core.mlf_logging import get_logger

_LOGGER = get_logger(__name__)

DEFAULT_BASE_CONTAINER = "/dev/shm/ml_flashpoint"
"""Default base container for checkpoint versions.

``/dev/shm`` is memory backed on the training nodes, which is what makes ML
Flashpoint saves fast. Point this elsewhere only if the node exposes a faster
node-local mount.
"""


@dataclasses.dataclass(frozen=True)
class MLFlashpointBridgeConfig:
    """ML Flashpoint settings for the Megatron Bridge adapter.

    Attributes:
        enabled: Whether ML Flashpoint handles non-persistent checkpoints. When
            False, the adapter delegates every operation to Megatron Bridge, which
            makes it a no-op wrapper. Useful for A/B experiments that keep the
            exact same launch command.
        base_container: Base container (directory) that holds one child container
            per checkpoint version. Must be node-local and is expected to be
            memory backed.
        async_save: Whether saves are scheduled asynchronously. Keeping this True
            is what removes the write from the training critical path.
        write_thread_count: Number of writer threads used per rank.
        initial_write_buffer_size_bytes: Initial size of each write buffer.
        use_optimized_save: Whether to use the optimized zero-copy tensor save.
        use_cached_ckpt_structure: Whether to reuse the save plan across steps.
            Only safe when the checkpoint structure is constant.
        use_fully_parallel_wrapper: Whether to wrap the save/load strategies so
            checkpoint data is spread evenly across ranks.
        keep_checkpoints_on_finalize: Whether to keep the ML Flashpoint container
            when training terminates normally. Off by default so buffers are
            released back to the node.
    """

    enabled: bool = True
    base_container: str = DEFAULT_BASE_CONTAINER
    async_save: bool = True
    write_thread_count: int = 1
    initial_write_buffer_size_bytes: int = DEFAULT_INITIAL_BUFFER_SIZE_BYTES
    use_optimized_save: bool = True
    use_cached_ckpt_structure: bool = False
    use_fully_parallel_wrapper: bool = True
    keep_checkpoints_on_finalize: bool = False

    def __post_init__(self):
        if not self.base_container:
            raise ValueError("base_container cannot be empty.")
        if self.write_thread_count < 1:
            raise ValueError(f"write_thread_count must be >= 1, got {self.write_thread_count}.")
        if self.initial_write_buffer_size_bytes <= 0:
            raise ValueError(
                f"initial_write_buffer_size_bytes must be > 0, got {self.initial_write_buffer_size_bytes}."
            )

    @classmethod
    def from_env(cls) -> "MLFlashpointBridgeConfig":
        """Builds a config from ``MLFLASHPOINT_*`` environment variables.

        Every field falls back to the dataclass default when its variable is
        unset. See :func:`ml_flashpoint.core.utils.get_env_var_prefix` for the
        prefix applied to each name below.

        Returns:
            The environment-derived configuration.
        """
        defaults = cls()
        return cls(
            enabled=utils.get_env_val_bool("BRIDGE_ENABLED", defaults.enabled),
            base_container=utils.get_env_val_str("BASE_CONTAINER", defaults.base_container),
            async_save=utils.get_env_val_bool("ASYNC_SAVE", defaults.async_save),
            write_thread_count=utils.get_env_val_int("WRITE_THREAD_COUNT", defaults.write_thread_count),
            initial_write_buffer_size_bytes=utils.get_env_val_int(
                "INITIAL_WRITE_BUFFER_SIZE_BYTES", defaults.initial_write_buffer_size_bytes
            ),
            use_optimized_save=utils.get_env_val_bool("USE_OPTIMIZED_SAVE", defaults.use_optimized_save),
            use_cached_ckpt_structure=utils.get_env_val_bool(
                "USE_CACHED_CKPT_STRUCTURE", defaults.use_cached_ckpt_structure
            ),
            use_fully_parallel_wrapper=utils.get_env_val_bool(
                "USE_FULLY_PARALLEL_WRAPPER", defaults.use_fully_parallel_wrapper
            ),
            keep_checkpoints_on_finalize=utils.get_env_val_bool(
                "KEEP_CHECKPOINTS_ON_FINALIZE", defaults.keep_checkpoints_on_finalize
            ),
        )


_CONFIGURED: Optional[MLFlashpointBridgeConfig] = None


def configure(config: MLFlashpointBridgeConfig) -> None:
    """Registers the config the adapter uses instead of reading the environment.

    Call this before Megatron Bridge builds its checkpoint manager, i.e. before
    ``megatron.bridge.training.setup`` (or, for NeMo RL, before the Megatron
    policy worker is initialized).

    Args:
        config: The configuration to use.
    """
    global _CONFIGURED
    _CONFIGURED = config
    _LOGGER.info("Registered ML Flashpoint Megatron Bridge config: %s", config)


def reset_configuration() -> None:
    """Drops a previously registered config so the environment is read again."""
    global _CONFIGURED
    _CONFIGURED = None


def get_config() -> MLFlashpointBridgeConfig:
    """Returns the registered config, or one derived from the environment.

    Returns:
        The effective configuration.
    """
    if _CONFIGURED is not None:
        return _CONFIGURED
    return MLFlashpointBridgeConfig.from_env()
