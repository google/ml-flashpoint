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
from typing import Callable, List, Set

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
        *,
        global_rank_getter: Callable[[], int],
        local_rank_getter: Callable[[], int],
        broadcast_object_list_func: Callable[..., None],
        all_gather_object_func: Callable[..., None],
        world_size_getter: Callable[[], int],
        recover_context: bool = False,
    ):
        """Initializes the NeMoMLFlashpointCheckpointLoader.

        Args:
            checkpoint_object_manager: The checkpoint object manager to use for
                reading data.
            replication_manager: The replication manager to use for retrieving
                missing checkpoint objects from peer nodes.
            global_rank_getter: A callable that returns the global rank.
            local_rank_getter: A callable that returns the node-local rank.
            broadcast_object_list_func: A callable with the same signature as
                ``torch.distributed.broadcast_object_list``.
            all_gather_object_func: A callable with the same signature as
                ``torch.distributed.all_gather_object``.
            world_size_getter: A callable that returns the world size.
            recover_context: Whether to recover the context directory if missing.
        """
        super().__init__(
            checkpoint_object_manager,
            replication_manager,
            global_rank_getter=global_rank_getter,
            local_rank_getter=local_rank_getter,
            broadcast_object_list_func=broadcast_object_list_func,
            all_gather_object_func=all_gather_object_func,
            world_size_getter=world_size_getter,
        )
        self._recover_context = recover_context

    @override
    def _get_extra_local_objects(self, container_path: Path) -> List[CheckpointObjectId]:
        local_objects = []
        if self._recover_context:
            context_path = container_path / "context"
            if context_path.is_dir():
                for root, _, files in os.walk(context_path):
                    for file in files:
                        local_objects.append(CheckpointObjectId(str(Path(root) / file)))
        return local_objects

    @override
    def _get_extra_needed_objects(
        self,
        checkpoint: CheckpointContainerId,
        available_objects_by_rank: dict[int, List[CheckpointObjectId]],
    ) -> Set[str]:
        extra_needed = set()
        if self._recover_context:
            # We assume that if a rank has the context dir, the content in the dir is complete.
            # We assume that these are the files needed by all the nodes.
            context_path = Path(checkpoint.data) / "context"
            for objs in available_objects_by_rank.values():
                for obj in objs:
                    try:
                        if Path(str(obj.data)).is_relative_to(context_path):
                            extra_needed.add(str(obj.data))
                    except ValueError:
                        # Path.is_relative_to raises ValueError if it's not relative to the path
                        pass
        return extra_needed
