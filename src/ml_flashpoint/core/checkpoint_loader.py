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

import abc
import fcntl
import io
import json
import logging
import os
import pickle
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import IO, List, Optional, Tuple, TypeVar, cast

import torch
import torch.distributed as dist
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint import Metadata
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.planner import (
    LoadItemType,
    LoadPlanner,
    ReadItem,
)
from torch.distributed.checkpoint.utils import _create_file_view
from typing_extensions import override

from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.core.defaults import (
    COMMON_STATE_FNAME,
    DIRTY_MARKER_SUFFIX,
    GLOBAL_RANK_PATTERN,
    MAGIC_BYTES,
    default_metadata_object_name,
)
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import get_num_of_nodes, log_execution_time
from ml_flashpoint.replication.replication_manager import ReplicationManager

M = TypeVar("M")

_LOGGER = get_logger(__name__)


class MLFlashpointCheckpointLoader(abc.ABC):
    """
    This is the main interface for loading checkpoints, providing functionality for the different
    stages in the load flow.

    See `DefaultMLFlashpointCheckpointLoader` as the canonical implementation, and the `ml_flashpoint.adapter`
    package for out-of-the-box framework integrations.
    """

    @abc.abstractmethod
    def read_metadata(
        self,
        checkpoint_id: CheckpointContainerId,
        object_name: str = default_metadata_object_name(),
    ) -> Metadata:
        """
        Reads and returns Metadata for the given checkpoint_id. The CheckpointObjectId to read and return is formed as
        `{checkpoint_id}/{object_name}`, where filename is optional and uses the default value when unspecified.

        Args:
            checkpoint_id: The CheckpointContainerId to read metadata for.
            object_name: The metadata object name. Optional, defaults to default_metadata_object_name().

        Returns:
            Metadata read from CheckpointObjectId(f"{checkpoint_id}/{object_name}").
        """
        pass

    @abc.abstractmethod
    def read_data(
        self,
        checkpoint_object_id: CheckpointObjectId,
        read_items: List[ReadItem],
        planner: LoadPlanner,
        storage_data: dict[int, _StorageInfo],
    ) -> None:
        """Reads data from the checkpoint object and loads it into the application state.

        Args:
            checkpoint_object_id: The ID of the checkpoint object to read from.
            read_items: A list of items to read from the checkpoint.
            planner: The load planner that coordinates the load process.
            storage_data: A dictionary containing storage information for the checkpoint.
        """
        pass

    @abc.abstractmethod
    def get_latest_complete_checkpoint(
        self, checkpoint_base_container: CheckpointContainerId
    ) -> Optional[CheckpointContainerId]:
        """Get the latest complete `CheckpointContainerId`.

        A checkpoint is considered complete if a directory `step-{step}_ckpt` exists
        and no corresponding `step-{step}_ckpt.*_unfinished` marker exists.

        Args:
            checkpoint_base_container: The base directory to search for checkpoints.

        Returns:
            A CheckpointContainerId for the latest complete checkpoint, or None if none are found.
        """
        pass


class DefaultMLFlashpointCheckpointLoader(MLFlashpointCheckpointLoader):
    """
    The default implementation of the MLFlashpointCheckpointLoader interface.
    """

    def __init__(
        self,
        checkpoint_object_manager: CheckpointObjectManager,
        replication_manager: ReplicationManager,
        recover_context: bool = False,
    ):
        """Initializes the DefaultMLFlashpointCheckpointLoader.

        Args:
            checkpoint_object_manager: The checkpoint object manager to use for
                reading data.
            replication_manager: The replication manager to use for retrieving
                missing checkpoint objects from peer nodes.
            recover_context: Whether to recover the context directory if missing.
        """
        self._checkpoint_object_manager = checkpoint_object_manager
        self._replication_manager = replication_manager
        self._recover_context = recover_context
        # Cache for available objects: CheckpointContainerId -> dict[object_path, list[rank]]
        self._available_objects_cache: dict[CheckpointContainerId, dict[str, List[int]]] = {}

    @override
    @log_execution_time(logger=_LOGGER, name="read_metadata")
    def read_metadata(
        self,
        checkpoint_id: CheckpointContainerId,
        object_name: str = default_metadata_object_name(),
    ) -> Metadata:
        metadata_path = Path(checkpoint_id.data) / object_name
        try:
            with open(metadata_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            _LOGGER.exception("Error reading metadata from '%s'", metadata_path)
            raise

    def read_tensor(self, buffer_slice: IO[bytes], req: ReadItem) -> torch.Tensor:
        """Read tensor from file slice.

        Args:
            buffer_slice (IO[bytes]): file slice to read from.
            req (ReadItem): read item.

        Returns:
            torch.Tensor: read tensor.
        """
        # Peek at magic bytes
        pos = buffer_slice.tell()
        magic = buffer_slice.read(len(MAGIC_BYTES))
        buffer_slice.seek(pos)

        if magic == MAGIC_BYTES:
            # New optimized format
            buffer_slice.seek(pos + len(MAGIC_BYTES))

            # Read header length
            len_bytes = buffer_slice.read(4)
            header_len = struct.unpack("<I", len_bytes)[0]

            # Read header
            json_bytes = buffer_slice.read(header_len)
            metadata = json.loads(json_bytes.decode("utf-8"))

            tensor_dtype = getattr(torch, metadata["dtype"])
            tensor_shape = metadata["shape"]
            data_bytes = buffer_slice.read()
            tensor = torch.frombuffer(data_bytes, dtype=tensor_dtype)
            tensor = tensor.reshape(tensor_shape)
        else:
            # Fallback to legacy torch.load
            tensor = cast(
                torch.Tensor,
                torch.load(cast(IO[bytes], buffer_slice), map_location="cpu", weights_only=True),
            )

        return narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)

    def _try_retrieve_object_if_missing(self, checkpoint_object_id: CheckpointObjectId) -> bool:
        """Attempts to retrieve a checkpoint object from peer nodes if it is missing locally.
        This method assume we have _available_objects_cache saved
        as this should be called after get_latest_complete_checkpoint, which populates _available_objects_cache.

        Args:
            checkpoint_object_id: The ID of the missing checkpoint object.

        Returns:
            True if the object was successfully retrieved or already exists, False otherwise.
        """
        if os.path.exists(checkpoint_object_id.data):
            _LOGGER.debug("Object '%s' already exists locally.", checkpoint_object_id.data)
            return True

        if not self._replication_manager:
            _LOGGER.error("ReplicationManager is not initialized.")
            return False

        current_container_id = checkpoint_object_id.get_parent()

        source_rank = -1
        found_source = False

        if current_container_id in self._available_objects_cache:
            locations = self._available_objects_cache[current_container_id]
            if checkpoint_object_id.data in locations:
                _LOGGER.debug(
                    "checkpoint_object_id.data: '%s' found in container '%s'",
                    checkpoint_object_id.data,
                    current_container_id,
                )
                sources = locations[checkpoint_object_id.data]
                if sources:
                    # TODO: Optimize the source rank selection to be more efficient,
                    # currently we just pick the first one.
                    source_rank = sources[0]
                    found_source = True

        if not found_source:
            return False

        _LOGGER.debug(
            "Object '%s' missing locally but found on rank %s. Attempting retrieval.",
            checkpoint_object_id.data,
            source_rank,
        )

        # TODO: use filelock lib.
        lock_path = f"{checkpoint_object_id.data}.lock"
        try:
            with open(lock_path, "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                try:
                    # Double check existence after acquiring lock
                    if os.path.exists(checkpoint_object_id.data):
                        _LOGGER.debug("Object '%s' appeared after acquiring lock.", checkpoint_object_id.data)
                        return True

                    _LOGGER.debug("Retrieving '%s' from rank %s", checkpoint_object_id.data, source_rank)
                    success = self._replication_manager.sync_bulk_retrieve(
                        source_global_rank=source_rank,
                        object_ids_to_retrieve=[checkpoint_object_id],
                        container_ids_to_retrieve=[],
                    )
                    if success:
                        _LOGGER.debug("Successfully retrieved '%s'", checkpoint_object_id.data)
                        return True
                    else:
                        _LOGGER.error("Failed to retrieve '%s'", checkpoint_object_id.data)
                        return False
                finally:
                    fcntl.flock(lock_file, fcntl.LOCK_UN)
        except Exception:
            _LOGGER.exception("Error during retrieval/locking for '%s'", checkpoint_object_id.data)
            return False

    @override
    @log_execution_time(logger=_LOGGER, name="read_data", level=logging.INFO)
    def read_data(
        self,
        checkpoint_object_id: CheckpointObjectId,
        read_items: List[ReadItem],
        planner: LoadPlanner,
        storage_data: dict[int, _StorageInfo],
    ) -> None:
        """
        If the checkpoint object is missing locally, attempt to retrieve
        it from a peer node with _try_retrieve_object_if_missing.
        """
        if not self._try_retrieve_object_if_missing(checkpoint_object_id):
            error_msg = f"Checkpoint object '{checkpoint_object_id.data}' does not exist on any node in the cluster"
            _LOGGER.error(error_msg)
            raise FileNotFoundError(error_msg)

        with self._checkpoint_object_manager.get_buffer(checkpoint_object_id) as stream:
            for req in read_items:
                item_md = storage_data[req.storage_index]
                buffer_slice = cast(IO[bytes], _create_file_view(stream, item_md.offset, item_md.length))
                if req.type == LoadItemType.BYTE_IO:
                    read_bytes = io.BytesIO(buffer_slice.read(item_md.length))
                    read_bytes.seek(0)
                    planner.load_bytes(req, read_bytes)
                else:
                    tensor = self.read_tensor(buffer_slice, req)
                    target_tensor = planner.resolve_tensor(req).detach()
                    assert target_tensor.size() == tensor.size(), (
                        f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                    )
                    target_tensor.copy_(tensor)
                    planner.commit_tensor(req, target_tensor)

    @override
    @log_execution_time(logger=_LOGGER, name="get_latest_complete_checkpoint", level=logging.INFO)
    def get_latest_complete_checkpoint(
        self, checkpoint_base_container: CheckpointContainerId
    ) -> Optional[CheckpointContainerId]:
        """
        Step 1: call get_candidate_checkpoints to get all existing checkpoint containers across
            all ranks as candidates and sorted in a descending order by step
        Step 2: traverse the candidate checkpoints and for each checkpoint, for each candidate:
            - call get_checkpoint_objects_by_rank to get all existing checkpoint objects cross
                all ranks of a candidate checkpoint
            - get retrieve plan for all ranks
                - make sure at least one rank has .metadata file
                - call _compute_retrieval_plan on that rank to get the retrieve plan for all ranks
                - broadcast the retrieve plan to all ranks
            - if it's retrievable, call retrieve_checkpoint for all ranks,
            else continue to the next candidate checkpoint
            - return the checkpoint container id of the latest complete checkpoint
        """
        # TODO: use global_rank_getter and local_rank_getter.
        rank = dist.get_rank()
        _LOGGER.debug(
            "Rank %s: Getting latest complete checkpoint for '%s'",
            rank,
            checkpoint_base_container,
        )

        candidate_checkpoints = self.get_candidate_checkpoints(checkpoint_base_container)
        _LOGGER.debug("Rank %s: Candidate checkpoints: '%s'", rank, candidate_checkpoints)
        if not candidate_checkpoints:
            _LOGGER.warning("No candidate checkpoints found.")
            return None

        for i, checkpoint in enumerate(candidate_checkpoints):
            available_objects_by_rank = self.get_checkpoint_objects_by_rank(checkpoint)
            _LOGGER.debug("Rank %s: Available objects by rank: '%s'", rank, available_objects_by_rank)

            planner_rank = -1
            found_metadata = False

            # Iterate in sorted order of ranks to ensure determinism
            for r in sorted(available_objects_by_rank.keys()):
                objs = available_objects_by_rank[r]
                for obj in objs:
                    if os.path.basename(obj.data) == default_metadata_object_name():
                        planner_rank = r
                        found_metadata = True
                        break
                if found_metadata:
                    break

            if found_metadata:
                _LOGGER.debug("Rank %s: Selected planner rank %s for '%s'", rank, planner_rank, checkpoint)
            else:
                _LOGGER.warning("Rank %s: No metadata found for '%s'", rank, checkpoint)
                continue

            retrieval_plan = None
            # Only the designated planner rank (the lowest rank id that has the metadata file)
            # computes the retrieval plan.
            if rank == planner_rank:
                retrieval_plan = self._compute_retrieval_plan(checkpoint, available_objects_by_rank)
            # Broadcast the retrieval plan to all ranks.
            plan_container = [retrieval_plan]
            dist.broadcast_object_list(plan_container, src=planner_rank)
            retrieval_plan = plan_container[0]

            if retrieval_plan is None:
                # If retrieval plan is None, it means the checkpoint is not viable.
                _LOGGER.warning("Rank %s: retrieval plan is None for '%s'. Not viable.", rank, checkpoint)
                continue

            if not retrieval_plan:
                # Empty dict means no retrieval needed, choose this checkpoint as the latest valid checkpoint.
                _LOGGER.debug("Rank %s: No retrieval needed for '%s'", rank, checkpoint)
                return checkpoint

            _LOGGER.debug("Rank %s: Retrieval plan: '%s'", rank, retrieval_plan)

            if self.retrieve_checkpoint(retrieval_plan):
                _LOGGER.debug("Successfully prepared checkpoint '%s' on all ranks.", checkpoint)
                return checkpoint
            else:
                _LOGGER.warning("Failed to retrieve all necessary objects for checkpoint '%s'.", checkpoint)
                continue

        _LOGGER.warning("Rank %s: No complete checkpoint found.", rank)
        return None

    def _compute_retrieval_plan(
        self,
        checkpoint: CheckpointContainerId,
        available_objects_by_rank: dict[int, List[CheckpointObjectId]],
    ) -> Optional[dict[int, List[Tuple[int, str]]]]:
        """Computes the retrieval plan.

        Args:
            checkpoint: The checkpoint container ID.
            available_objects_by_rank: Map of rank to available objects on that rank.

        Returns:
            A retrieval plan or None if the checkpoint is not viable. Empty dict means no retrieval needed.
        """
        try:
            metadata = self.read_metadata(checkpoint)
        except Exception:
            _LOGGER.warning("Failed to read metadata for '%s'", checkpoint)
            return None

        storage_data = metadata.storage_data
        if storage_data is None:
            return None

        all_needed_checkpoint_objects_by_rank: dict[int, set[str]] = defaultdict(set)
        for _, storage_info in storage_data.items():
            match = GLOBAL_RANK_PATTERN.search(storage_info.relative_path)
            if match:
                global_rank = int(match.group(1))
                full_path = CheckpointObjectId.from_container(checkpoint, storage_info.relative_path)
                all_needed_checkpoint_objects_by_rank[global_rank].add(str(full_path))

        # Identify objects needed by the first rank of each node (local_rank 0).
        # This includes common state files, metadata, and optionally context files.
        objects_needed_by_local_rank_0 = set()
        objects_needed_by_local_rank_0.add(str(CheckpointObjectId.from_container(checkpoint, COMMON_STATE_FNAME)))
        objects_needed_by_local_rank_0.add(
            str(CheckpointObjectId.from_container(checkpoint, default_metadata_object_name()))
        )

        if self._recover_context:
            # We assume that if a rank has the context dir, the content in the dir is complete.
            # We assume that are the files needed by all the nodes.
            context_path = Path(checkpoint.data) / "context"
            for objs in available_objects_by_rank.values():
                for obj in objs:
                    if Path(obj.data).parent == context_path:
                        objects_needed_by_local_rank_0.add(str(obj.data))

        world_size = dist.get_world_size()
        num_nodes = get_num_of_nodes()
        ranks_per_node = world_size // num_nodes

        for rank in range(world_size):
            # Only local_node 0 needs to retrieve these common objects
            if rank % ranks_per_node == 0:
                all_needed_checkpoint_objects_by_rank[rank].update(objects_needed_by_local_rank_0)

        # Build reverse map for object locations
        object_locations = defaultdict(list)
        for rank, objects in available_objects_by_rank.items():
            for obj in objects:
                obj_str = str(obj)
                object_locations[obj_str].append(rank)

        retrieval_plan = {}
        for target_rank, needed_objs in all_needed_checkpoint_objects_by_rank.items():
            already_has = {str(o) for o in available_objects_by_rank.get(target_rank, [])}
            missing = needed_objs - already_has

            if missing:
                retrieval_plan[target_rank] = []
                for missing_obj in missing:
                    sources = object_locations.get(missing_obj)
                    if not sources:
                        # If it's missing globally, we can't retrieve it.
                        # This will result in returning None for the plan, effectively skipping this checkpoint.
                        _LOGGER.warning("Object '%s' is missing globally (needed by rank %s)", missing_obj, target_rank)
                        return None
                    # TODO: Optimize the source rank selection to be more efficient
                    # default to the first source rank for now.
                    retrieval_plan[target_rank].append((sources[0], missing_obj))

        return retrieval_plan

    @log_execution_time(logger=_LOGGER, name="get_candidate_checkpoints")
    def get_candidate_checkpoints(
        self, checkpoint_base_container: CheckpointContainerId
    ) -> List[CheckpointContainerId]:
        """Gathers all apparently finished checkpoint containers from all nodes and returns the sorted union.

        This method first finds all locally available checkpoints, then gathers the lists from all nodes
        in the distributed environment, computes the intersection of these lists, and returns a single sorted
        list of candidate checkpoints.

        Args:
            checkpoint_base_container: The base container ID to search for checkpoints.

        Returns:
            A sorted list of candidate checkpoint container IDs in descending order.
        """
        _LOGGER.debug("Getting candidate checkpoints for '%s'", checkpoint_base_container)

        # Scan locally only on the first rank of each node
        base_path = Path(checkpoint_base_container.data)
        rank = dist.get_rank()
        local_rank = dist.get_node_local_rank()

        local_candidate_ckpt_ids = []

        _LOGGER.debug("Rank %s (Local Rank %s): Checking base path: '%s'", rank, local_rank, base_path)
        if base_path.is_dir():
            ckpt_pattern = re.compile(r"(step-\d+_ckpt)")
            potential_ckpts = set()
            unfinished_ckpts = set()

            for entry in os.listdir(base_path):
                ckpt_match = ckpt_pattern.match(entry)
                if ckpt_match:
                    if not entry.endswith(DIRTY_MARKER_SUFFIX):
                        potential_ckpts.add(entry)
                    else:
                        ckpt_id = ckpt_match.group(1)
                        _LOGGER.debug("Found unfinished marker: '%s'", entry)
                        unfinished_ckpts.add(ckpt_id)

            local_candidate_ckpt_ids = [str(base_path / p) for p in (potential_ckpts - unfinished_ckpts)]
        else:
            _LOGGER.debug("Rank %s: Base path '%s' is not a directory or does not exist.", rank, base_path)

        all_checkpoint_container_path_lists = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_checkpoint_container_path_lists, local_candidate_ckpt_ids)
        _LOGGER.debug(
            "Rank %s: Gathered checkpoint container paths from all ranks: '%s'",
            rank,
            all_checkpoint_container_path_lists,
        )

        # Filter out None values and use set.intersection to gather all candidates
        valid_path_lists = [set(paths) for paths in all_checkpoint_container_path_lists if paths]
        if not valid_path_lists:
            _LOGGER.debug("No valid checkpoint container paths found.")
            return []
        else:
            intersection_of_checkpoint_containers = set.intersection(*valid_path_lists)

        _LOGGER.debug(
            "Rank %s: Intersection of checkpoint containers: '%s'", rank, intersection_of_checkpoint_containers
        )

        candidate_ckpt_ids = [CheckpointContainerId(p) for p in intersection_of_checkpoint_containers]

        return sorted(
            candidate_ckpt_ids,
            key=lambda cid: CheckpointContainerId.parse_version_container_step(os.path.basename(cid.data)) or -1,
            reverse=True,
        )

    @log_execution_time(logger=_LOGGER, name="get_checkpoint_objects_by_rank")
    def get_checkpoint_objects_by_rank(
        self, checkpoint_container_id: CheckpointContainerId
    ) -> dict[int, List[CheckpointObjectId]]:
        """Gathers all available checkpoint objects from all nodes for a given checkpoint.

        This method directly inspects the checkpoint container directory on each node
        to identify checkpoint object files.

        Args:
            checkpoint_container_id: The ID of the checkpoint container to inspect.

        Returns:
            A dictionary mapping each node's local rank 0 to a list of
            `CheckpointObjectId`s available on that node.
        """
        container_path = Path(checkpoint_container_id.data)
        local_objects = []
        if not container_path.is_dir():
            _LOGGER.debug(
                "Checkpoint container path '%s' is not a directory. Returning empty list.",
                container_path,
            )
        else:
            for entry in os.listdir(container_path):
                local_objects.append(entry)

            if self._recover_context:
                context_path = container_path / "context"
                if context_path.is_dir():
                    for entry in os.listdir(context_path):
                        local_objects.append(os.path.join("context", entry))

        all_objects_by_rank_paths = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_objects_by_rank_paths, local_objects)

        result = {}
        object_locations = defaultdict(list)
        if all_objects_by_rank_paths:
            for rank, objects in enumerate(all_objects_by_rank_paths):
                if objects:
                    # Convert filenames to full paths and then to CheckpointObjectId
                    full_paths = [str(container_path / obj) for obj in objects]
                    checkpoint_objects = [CheckpointObjectId(p) for p in full_paths]
                    result[rank] = checkpoint_objects
                    for obj in checkpoint_objects:
                        object_locations[obj.data].append(rank)
                else:
                    result[rank] = []

        self._available_objects_cache[checkpoint_container_id] = object_locations
        _LOGGER.debug("Available objects for '%s': '%s'", checkpoint_container_id, object_locations)
        _LOGGER.debug("_available_objects_cache '%s'", self._available_objects_cache)
        return result

    @log_execution_time(logger=_LOGGER, name="retrieve_checkpoint")
    def retrieve_checkpoint(
        self,
        retrieval_plan: dict[int, List[Tuple[int, str]]],
    ) -> bool:
        """Retrieves missing checkpoint objects based on the retrieval plan.

        Args:
            retrieval_plan: A dictionary mapping Rank -> List of (SourceRank, ObjectPath).
                            If empty for this rank, no retrieval is needed.
        """

        rank = dist.get_rank()
        all_success = True

        # Only proceed with retrieval if we have items to retrieve
        if retrieval_plan and rank in retrieval_plan:
            items_to_retrieve = retrieval_plan[rank]
            if items_to_retrieve:
                _LOGGER.debug("Retrieving %d objects for rank %s", len(items_to_retrieve), rank)

                # Group by source rank
                objects_to_retrieve_by_source: dict[int, List[CheckpointObjectId]] = {}
                for source_rank, obj_path in items_to_retrieve:
                    if source_rank not in objects_to_retrieve_by_source:
                        objects_to_retrieve_by_source[source_rank] = []
                    objects_to_retrieve_by_source[source_rank].append(CheckpointObjectId(obj_path))

                _LOGGER.debug("Retrieving %d objects for rank %s", len(objects_to_retrieve_by_source), rank)

                for source_rank, objects_to_retrieve in objects_to_retrieve_by_source.items():
                    _LOGGER.debug(
                        "Retrieving %d objects for rank %s from source rank %s",
                        len(objects_to_retrieve),
                        rank,
                        source_rank,
                    )
                    success = self._replication_manager.sync_bulk_retrieve(
                        source_global_rank=source_rank,
                        object_ids_to_retrieve=objects_to_retrieve,
                        container_ids_to_retrieve=[],  # Not used yet
                    )
                    if not success:
                        _LOGGER.error("Failed to retrieve objects from rank %s", source_rank)
                        all_success = False

        # Gather success status from all ranks
        _LOGGER.debug("Gathering success status from all ranks")
        all_success_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(all_success_list, all_success)
        _LOGGER.debug("All success list: '%s'", all_success_list)
        return all(all_success_list)
