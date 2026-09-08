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

"""Advertises ML Flashpoint checkpoints to Megatron Bridge's resume decision."""

import json
import os
from typing import Optional

import torch.distributed as dist

from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.checkpoint_loader import MLFlashpointCheckpointLoader
from ml_flashpoint.core.mlf_logging import get_logger

_LOGGER = get_logger(__name__)

NO_CHECKPOINT = -1
"""Sentinel Megatron Bridge uses for "no local checkpoint exists"."""


class MLFlashpointLocalCheckpointIndex:
    """Duck-types the part of NVRx's ``LocalCheckpointManager`` that Bridge reads.

    Megatron Bridge decides whether to attempt a resume in
    ``megatron.bridge.training.setup._should_load_checkpoint``, which looks for a
    ``local_checkpoint_manager`` in the checkpoint manager's context and calls
    ``find_latest()`` on it. Without this shim a run whose only checkpoint is an
    ML Flashpoint one would start from scratch.

    Only ``find_latest`` and ``local_ckpt_dir`` are implemented: the adapter's own
    checkpoint manager performs the load, so Bridge's local load path must never
    be reached.
    """

    def __init__(
        self,
        base_container: CheckpointContainerId,
        checkpoint_loader: MLFlashpointCheckpointLoader,
    ):
        """Initializes the index.

        Args:
            base_container: The base container holding checkpoint versions.
            checkpoint_loader: The loader used to find recoverable checkpoints.
        """
        self._base_container = base_container
        self._checkpoint_loader = checkpoint_loader
        self._latest: Optional[CheckpointContainerId] = None
        self._resolved = False
        self._disabled = False

    @property
    def local_ckpt_dir(self) -> str:
        """The directory Bridge would use as the local checkpoint root."""
        return str(self._base_container)

    @property
    def latest_container(self) -> Optional[CheckpointContainerId]:
        """The resolved latest complete container, if discovery already ran."""
        return self._latest

    def disable(self) -> None:
        """Makes ``find_latest`` report no checkpoint from now on.

        Called before delegating to Megatron Bridge so that Bridge does not try to
        read an ML Flashpoint container through its own local-checkpoint path,
        which expects an NVRx ``MCoreTensorAwareStateDict`` container.
        """
        self._disabled = True

    def find_latest(self) -> int:
        """Returns the step of the latest complete ML Flashpoint checkpoint.

        Discovery is collective (it gathers per-rank object inventories) and is
        therefore performed at most once, with the result cached.

        Returns:
            The step number, or ``NO_CHECKPOINT`` when nothing is recoverable.
        """
        container = self.resolve_latest_container()
        if container is None:
            return NO_CHECKPOINT
        step = CheckpointContainerId.parse_version_container_step(os.path.basename(str(container)))
        return NO_CHECKPOINT if step is None else step

    def resolve_latest_container(self) -> Optional[CheckpointContainerId]:
        """Finds (once) the latest complete checkpoint container.

        Returns:
            The container, or None when nothing is recoverable.
        """
        if self._disabled:
            return None
        if self._resolved:
            return self._latest

        self._resolved = True
        try:
            self._latest = self._checkpoint_loader.get_latest_complete_checkpoint(self._base_container)
        except Exception:
            _LOGGER.exception("Failed to discover ML Flashpoint checkpoints under '%s'.", self._base_container)
            self._latest = None

        if self._latest is not None:
            _ensure_megatron_metadata_stub(self._latest)
        return self._latest

    def invalidate(self) -> None:
        """Forces the next ``find_latest`` call to rediscover."""
        self._resolved = False
        self._latest = None


def _ensure_megatron_metadata_stub(container: CheckpointContainerId) -> None:
    """Writes the ``metadata.json`` stub Megatron's loader validates.

    ML Flashpoint stores tensor data in shared-memory buffers rather than the
    ``.distcp`` layout, but ``dist_checkpointing.load`` still validates a backend
    marker file before dispatching to the strategy. The checks against this stub
    are no-ops, and the save path writes it too; this covers a container that was
    replicated from a peer node without the file.

    Args:
        container: The checkpoint container to stub.
    """
    try:
        if dist.is_initialized() and dist.get_node_local_rank() != 0:
            return
        metadata_path = os.path.join(str(container), "metadata.json")
        if os.path.exists(metadata_path):
            return
        with open(metadata_path, "w") as f:
            json.dump({"sharded_backend": ""}, f)
        _LOGGER.debug("Wrote Megatron metadata stub at '%s'", metadata_path)
    except Exception:
        _LOGGER.exception("Failed to write the Megatron metadata stub for '%s'.", container)
