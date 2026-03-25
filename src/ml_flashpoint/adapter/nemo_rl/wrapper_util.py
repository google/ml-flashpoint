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

from typing import Any

from ml_flashpoint.adapter.megatron.save_strategies import MLFlashpointMegatronAsyncSaveStrategy
from ml_flashpoint.adapter.nemo_rl.checkpoint_manager import MLFlashpointRLCheckpointManager
from ml_flashpoint.core.checkpoint_loader import MLFlashpointCheckpointLoader
from ml_flashpoint.core.mlf_logging import get_logger

_LOGGER = get_logger(__name__)


def wrap_rl_components_with_mlflashpoint(
    checkpointer: Any,
    flashpoint_base_container: str,
    standard_save_period: int,
    save_strategy: MLFlashpointMegatronAsyncSaveStrategy,
    checkpoint_loader: MLFlashpointCheckpointLoader,
) -> Any:
    """Wraps a NeMo/RL CheckpointManager and Policy with ML Flashpoint logic.

    This utility makes it easy to inject ML Flashpoint complementary checkpointing
    directly into NeMo/RL algorithm recipes without altering upstream code. It swaps
    the standard checkpointer with a dual-manager that understands frequent ML Flashpoint
    in-memory saves versus sparse standard disk saves.

    Args:
        checkpointer: The original NeMo/RL CheckpointManager.
        flashpoint_base_container: Base namespace string for ML Flashpoint saves.
        standard_save_period: The step frequency for taking standard permanent saves.
        checkpoint_loader: The MLFlashpointCheckpointLoader for resolving latest MLF saves.
        save_strategy: The MLFlashpointMegatronAsyncSaveStrategy instance.

    Returns:
        MLFlashpointRLCheckpointManager: The wrapped checkpointer to pass to your algorithm loop.
    """
    _LOGGER.info(
        "Wrapping NeMo/RL checkpointer and policy with ML Flashpoint Dual CheckpointManager. "
        f"Standard save config period: {standard_save_period}."
    )

    return MLFlashpointRLCheckpointManager(
        base_checkpointer=checkpointer,
        flashpoint_base_container=flashpoint_base_container,
        standard_save_period=standard_save_period,
        save_strategy=save_strategy,
        checkpoint_loader=checkpoint_loader,
    )
