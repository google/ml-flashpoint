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

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch.distributed as dist
from nemo.lightning import AutoResume
from nemo.lightning import io as nemo_lightning_io
from typing_extensions import override

from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.checkpoint_loader import MLFlashpointCheckpointLoader
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import log_execution_time

_LOGGER = get_logger(__name__)


@dataclass(kw_only=True)
class MLFlashpointAutoResume(AutoResume):
    """
    MLFlashpoint-specific AutoResume implementation that prioritizes local MLFlashpoint checkpoints to recover from.

    This class requires a checkpoint_loader and checkpoint_base_container to be provided. Additional parent class
    attributes may also be provided e.g.:
        MLFlashpointAutoResume(
            checkpoint_loader=loader, # Required
            checkpoint_base_container=base_container, # Required
            resume_if_exists=True, # Strongly recommended
            resume_ignore_no_checkpoint=True, # Strongly recommended
            restore_config=config, # Optional parent class attribute
        )

    This class will always try to find an ML Flashpoint checkpoint to recover from first, and if none found,
    delegates to the parent class to try to find a regular checkpoint.

    Attributes:
        checkpoint_base_container (CheckpointContainerId): The base container path to store and find checkpoint
            containers.
        checkpoint_loader (MLFlashpointCheckpointLoader): The loader to leverage for identifying candidate recovery
            checkpoints and selecting a suitable one.
    """

    checkpoint_base_container: CheckpointContainerId
    checkpoint_loader: MLFlashpointCheckpointLoader
    _checkpoint_path_cache: Optional[Path] = field(default=None, init=False)
    _flashpoint_resolved: bool = field(default=False, init=False)

    @override
    @log_execution_time(logger=_LOGGER, name="MLFlashpointAutoResume._find_trainer_ckpt_path", level=logging.INFO)
    def _find_trainer_ckpt_path(self) -> Optional[Path]:
        if self._checkpoint_path_cache is not None:
            return self._checkpoint_path_cache

        if not self._flashpoint_resolved:
            self._flashpoint_resolved = True
            local_container = self.checkpoint_loader.get_latest_complete_checkpoint(self.checkpoint_base_container)
            _LOGGER.info("Latest complete checkpoint: '%s'", local_container)

            if local_container is not None:
                # check each node's latest valid checkpoint dir has empty metadata.json, if not, create one
                local_rank = dist.get_node_local_rank()
                if local_rank == 0:
                    metadata_json_file = os.path.join(local_container.data, "metadata.json")
                    _LOGGER.debug("Rank %s: Checking metadata file '%s'", local_rank, metadata_json_file)
                    if not os.path.exists(metadata_json_file):
                        metadata = {"sharded_backend": ""}
                        with open(metadata_json_file, "w") as f:
                            json.dump(metadata, f)
                self._checkpoint_path_cache = Path(local_container.data)
                return self._checkpoint_path_cache

        return super()._find_trainer_ckpt_path()

    @override
    def get_trainer_ckpt_path(self, model: Optional[nemo_lightning_io.ConnectorMixin] = None) -> Optional[Path]:
        return self._find_trainer_ckpt_path()
