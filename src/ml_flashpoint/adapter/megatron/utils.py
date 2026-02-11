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

import os
from typing import Any, Optional, Union

import torch
from megatron.core.dist_checkpointing import state_dict_utils as mcore_state_dict_utils
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.core.dist_checkpointing.strategies.common import COMMON_STATE_FNAME

from ml_flashpoint.core.mlf_logging import get_logger

_LOGGER = get_logger(__name__)


def save_local_aware_megatron_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_dir: Union[str, torch.Path],
    save_strategy,
    async_save: bool = True,
) -> Optional[AsyncRequest]:
    """Saves a checkpoint with local-aware common state handling.

    This function mimics the CommonStrategy logic from Megatron's dist_checkpointing.save(),
    but with a key difference: it saves common data on each node (via local rank 0)
    rather than solely on the coordinator node (global rank 0).

    This is necessary for local checkpointing where each node needs its own copy
    of the common state for fast recovery.

    Args:
        checkpoint: The checkpoint dictionary to save.
        checkpoint_dir: The directory path to save the checkpoint to.
        save_strategy: The save strategy instance with async_save() and save() methods.
            Typically MLFlashpointMegatronAsyncSaveStrategy.
        async_save: Whether to save asynchronously. Defaults to True.

    Returns:
        An AsyncRequest if async_save is True and save succeeds, None otherwise.
        Returns None on save failure (exception is logged).
    """
    # Split common and sharded state
    sharded_state_dict, common_state_dict = mcore_state_dict_utils.save_preprocess(checkpoint)

    # Save common state on each node (local rank 0)
    if torch.distributed.get_node_local_rank() == 0:
        _LOGGER.debug("Saving common_state_dict...")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(common_state_dict, os.path.join(checkpoint_dir, COMMON_STATE_FNAME))

    # Execute save strategy
    try:
        if async_save:
            return save_strategy.async_save(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=checkpoint_dir,
            )
        else:
            save_strategy.save(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=checkpoint_dir,
            )
            return None
    except Exception:
        _LOGGER.exception("Failed to save ML Flashpoint checkpoint. Skipping saving and continuing.")
        return None
