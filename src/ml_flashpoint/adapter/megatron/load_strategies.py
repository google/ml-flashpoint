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

import io
import logging
from pathlib import Path
from typing import Union

import torch
from megatron.core.dist_checkpointing.mapping import (
    ShardedObject,
    ShardedStateDict,
    ShardedTensor,
    StateDict,
)
from megatron.core.dist_checkpointing.strategies.base import LoadShardedStrategy
from megatron.core.dist_checkpointing.strategies.torch import (
    MCoreLoadPlanner,
    _replace_sharded_keys_with_state_dict_keys,
    _replace_state_dict_keys_with_sharded_keys,
    _restore_dict_types,
    _unwrap_pyt_sharded_tensor,
    mcore_to_pyt_state_dict,
)
from torch.distributed import checkpoint as torch_dist_checkpoint
from torch.distributed._shard.sharded_tensor import ShardedTensor as TorchShardedTensor
from torch.distributed.checkpoint import (
    BytesStorageMetadata,
    Metadata,
    TensorStorageMetadata,
)
from typing_extensions import override

from ml_flashpoint.adapter.pytorch.memory_storage_reader import MemoryStorageReader
from ml_flashpoint.core.checkpoint_loader import (
    CheckpointContainerId,
    MLFlashpointCheckpointLoader,
)
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import log_execution_time
from ml_flashpoint.replication.replication_manager import ReplicationManager

_LOGGER = get_logger(__name__)


class MLFlashpointMegatronLoadStrategy(LoadShardedStrategy):
    """Megatron Checkpoint Load Strategy using ML Flashpoint.

    This strategy leverages the ML Flashpoint library to load sharded model
    checkpoints. It interacts with a provided or default
    MLFlashpointCheckpointLoader instance to handle the actual loading operations.
    """

    def __init__(
        self,
        replication_manager: ReplicationManager,
        checkpoint_loader: MLFlashpointCheckpointLoader,
    ):
        """Initializes the load strategy.

        Args:
            replication_manager: The replication manager to use.
            checkpoint_loader: The loader instance to delegate load
                operations to.
        """
        super().__init__()
        self.checkpoint_loader = checkpoint_loader
        self._replication_manager = replication_manager

    @override
    @log_execution_time(logger=_LOGGER, name="load", level=logging.INFO)
    def load(self, sharded_state_dict: ShardedStateDict, checkpoint_dir: Union[str, Path]) -> StateDict:
        orig_sharded_state_dict = sharded_state_dict
        (sharded_state_dict, flat_mapping, rename_mapping) = _replace_state_dict_keys_with_sharded_keys(
            sharded_state_dict
        )
        pyt_state_dict = mcore_to_pyt_state_dict(sharded_state_dict, is_loading=True)
        storage_reader = MemoryStorageReader(
            path=checkpoint_dir,
            checkpoint_loader=self.checkpoint_loader,
        )
        # Must use Megatron's planner to satisfy various expectations when training.
        planner = MCoreLoadPlanner()
        torch_dist_checkpoint.load(state_dict=pyt_state_dict, storage_reader=storage_reader, planner=planner)

        mlf_state_dict: dict[str, Union[TorchShardedTensor, list[io.BytesIO]]] = {
            k: v if not isinstance(v, TorchShardedTensor) else _unwrap_pyt_sharded_tensor(v)
            for k, v in pyt_state_dict.items()
        }
        mlf_state_dict = _replace_sharded_keys_with_state_dict_keys(mlf_state_dict, flat_mapping, rename_mapping)
        # Need to restore dict key types to handle str<->int conversions for later merging/processing.
        _restore_dict_types(mlf_state_dict, orig_sharded_state_dict)
        return mlf_state_dict

    @override
    @log_execution_time(logger=_LOGGER, name="load_tensors_metadata", level=logging.INFO)
    def load_tensors_metadata(self, checkpoint_dir: Union[str, Path], metadata: Metadata = None) -> StateDict:
        if metadata is None:
            metadata = self.checkpoint_loader.read_metadata(CheckpointContainerId(checkpoint_dir))
            if metadata is None or not hasattr(metadata, "state_dict_metadata"):
                raise RuntimeError(f"Failed to load valid metadata from {checkpoint_dir}")

        sharded_metadata = {}
        for k, tp in metadata.state_dict_metadata.items():
            if not isinstance(tp, TensorStorageMetadata):
                continue  # load only tensors
            sharded_metadata[k] = ShardedTensor.from_rank_offsets(
                k, torch.empty(tp.size, **tp.properties.__dict__, device="meta")
            ).without_data()

        return sharded_metadata

    @override
    @log_execution_time(logger=_LOGGER, name="load_sharded_metadata", level=logging.INFO)
    def load_sharded_metadata(self, checkpoint_dir: Union[str, Path]) -> ShardedStateDict:
        # Must be implemented if can_handle_sharded_objects is True.
        # Otherwise, superclass delegates to load_tensors_metadata.
        metadata = self.checkpoint_loader.read_metadata(CheckpointContainerId(checkpoint_dir))
        if metadata is None or not hasattr(metadata, "state_dict_metadata"):
            raise RuntimeError(f"Failed to load valid metadata from {checkpoint_dir}")
        sharded_metadata = {}
        for metadata_key, storage_metadata in metadata.state_dict_metadata.items():
            if not isinstance(storage_metadata, BytesStorageMetadata):
                continue
            sh_obj = ShardedObject.empty_from_unique_key(metadata_key)
            sharded_metadata[sh_obj.unique_key] = sh_obj

        sharded_metadata.update(self.load_tensors_metadata(checkpoint_dir, metadata))
        return sharded_metadata

    @override
    def check_backend_compatibility(self, loaded_backend):
        # Backward compatibility is not applicable to MLFlashpoint, since it is only
        # used to load within the same job instance, which is running the same version of code.
        # Hence, we always assume compatibility.
        pass

    @override
    def check_version_compatibility(self, loaded_version):
        # Backward compatibility is not applicable to MLFlashpoint, since it is only
        # used to load/recover within the same job instance, which is running the same version of code.
        # Hence, we always assume compatibility.
        pass

    @property
    def can_handle_sharded_objects(self) -> bool:
        # TODO: This is likely needed to be True for proper distributed checkpointing,
        #  which means we must also implement load_sharded_metadata.
        return True
