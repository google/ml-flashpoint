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

"""ML Flashpoint adapter for Megatron Bridge."""

from typing import Optional

from ml_flashpoint.adapter.megatron_bridge.checkpoint_manager import (
    MLFlashpointBridgeCheckpointManager as MLFlashpointBridgeCheckpointManager,
)
from ml_flashpoint.adapter.megatron_bridge.config import (
    DEFAULT_BASE_CONTAINER as DEFAULT_BASE_CONTAINER,
)
from ml_flashpoint.adapter.megatron_bridge.config import (
    MLFlashpointBridgeConfig as MLFlashpointBridgeConfig,
)
from ml_flashpoint.adapter.megatron_bridge.config import configure as configure
from ml_flashpoint.adapter.megatron_bridge.config import get_config as get_config
from ml_flashpoint.adapter.megatron_bridge.config import reset_configuration as reset_configuration
from ml_flashpoint.adapter.megatron_bridge.runtime import (
    MLFlashpointBridgeRuntime as MLFlashpointBridgeRuntime,
)
from ml_flashpoint.adapter.megatron_bridge.runtime import get_runtime as get_runtime
from ml_flashpoint.adapter.megatron_bridge.runtime import shutdown_runtime as shutdown_runtime
from ml_flashpoint.core.mlf_logging import get_logger

_LOGGER = get_logger(__name__)

CUSTOM_MANAGER_CLASS = "ml_flashpoint.adapter.megatron_bridge.MLFlashpointBridgeCheckpointManager"
"""Value to assign to ``CheckpointConfig.custom_manager_class``."""

_ALLOWLIST_PREFIX = "ml_flashpoint"


def register_with_megatron_bridge() -> None:
    """Allows Megatron Bridge to import the ML Flashpoint checkpoint manager.

    Bridge validates ``custom_manager_class`` against an import allowlist that
    only covers a fixed set of prefixes, so ``ml_flashpoint`` has to be added
    before the manager can be instantiated from config. Idempotent.
    """
    try:
        from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix
    except ImportError:
        _LOGGER.warning(
            "Could not import megatron.bridge.utils.instantiate_utils.register_allowed_target_prefix; "
            "this Megatron Bridge version may reject custom_manager_class='%s'.",
            CUSTOM_MANAGER_CLASS,
        )
        return
    register_allowed_target_prefix(_ALLOWLIST_PREFIX)
    _LOGGER.debug("Registered '%s' as an allowed Megatron Bridge target prefix.", _ALLOWLIST_PREFIX)


def enable(
    checkpoint_config,
    non_persistent_save_interval: int,
    mlf_config: Optional[MLFlashpointBridgeConfig] = None,
) -> None:
    """Points a Megatron Bridge ``CheckpointConfig`` at ML Flashpoint.

    Mutates ``checkpoint_config`` in place so that Bridge saves a fast,
    node-local ML Flashpoint checkpoint every ``non_persistent_save_interval``
    steps, while durable checkpoints keep going wherever ``save`` points, on the
    existing ``save_interval`` cadence.

    Bridge only takes the non-persistent branch on steps that are *not* also
    durable-checkpoint steps, so the two cadences do not collide.

    Args:
        checkpoint_config: The Bridge ``CheckpointConfig`` to modify.
        non_persistent_save_interval: How often, in steps, to write an ML
            Flashpoint checkpoint. Must be positive.
        mlf_config: ML Flashpoint settings to register. When omitted, settings
            are read from ``MLFLASHPOINT_*`` environment variables.

    Raises:
        ValueError: If ``non_persistent_save_interval`` is not positive.
    """
    if non_persistent_save_interval < 1:
        raise ValueError(
            f"non_persistent_save_interval must be a positive integer, got {non_persistent_save_interval}."
        )
    if mlf_config is not None:
        configure(mlf_config)

    register_with_megatron_bridge()
    checkpoint_config.custom_manager_class = CUSTOM_MANAGER_CLASS
    checkpoint_config.non_persistent_ckpt_type = "local"
    checkpoint_config.non_persistent_save_interval = non_persistent_save_interval
    _LOGGER.info(
        "Enabled ML Flashpoint for Megatron Bridge: non_persistent_save_interval=%d, config=%s",
        non_persistent_save_interval,
        get_config(),
    )
