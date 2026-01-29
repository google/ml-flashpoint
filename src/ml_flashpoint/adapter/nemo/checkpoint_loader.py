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
from pathlib import Path
from typing import List, Set

from typing_extensions import override

from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.checkpoint_loader import DefaultMLFlashpointCheckpointLoader
from ml_flashpoint.replication.replication_manager import ReplicationManager


class NeMoMLFlashpointCheckpointLoader(DefaultMLFlashpointCheckpointLoader):
    """
    NeMo-specific implementation of the MLFlashpointCheckpointLoader interface.
    """

    def __init__(
        self,
        checkpoint_object_manager: CheckpointObjectManager,
        replication_manager: ReplicationManager,
        recover_context: bool = False,
    ):
        """Initializes the NeMoMLFlashpointCheckpointLoader.

        Args:
            checkpoint_object_manager: The checkpoint object manager to use for
                reading data.
            replication_manager: The replication manager to use for retrieving
                missing checkpoint objects from peer nodes.
            recover_context: Whether to recover the context directory if missing.
        """
        super().__init__(checkpoint_object_manager, replication_manager)
        self._recover_context = recover_context

    @override
    def _get_extra_local_objects(self, container_path: Path) -> List[str]:
        """Returns extra local objects to include, specifically context files."""
        local_objects = []
        if self._recover_context:
            context_path = container_path / "context"
            if context_path.is_dir():
                for entry in os.listdir(context_path):
                    local_objects.append(os.path.join("context", entry))
        return local_objects

    @override
    def _get_extra_needed_objects(
        self,
        checkpoint: CheckpointContainerId,
        available_objects_by_rank: dict[int, List[CheckpointObjectId]],
    ) -> Set[str]:
        """Returns extra needed objects for local rank 0, specifically context files."""
        extra_needed = set()
        if self._recover_context:
            # We assume that if a rank has the context dir, the content in the dir is complete.
            # We assume that are the files needed by all the nodes.
            context_path = Path(checkpoint.data) / "context"
            for objs in available_objects_by_rank.values():
                for obj in objs:
                    if Path(obj.data).parent == context_path:
                        extra_needed.add(str(obj.data))
        return extra_needed
