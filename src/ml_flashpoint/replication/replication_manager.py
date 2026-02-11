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
import concurrent.futures
import logging
import socket
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.distributed as dist
from typing_extensions import override

from ml_flashpoint.checkpoint_object_manager.buffer_io import BufferIO
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import (
    CheckpointObjectManager,
)
from ml_flashpoint.core import utils
from ml_flashpoint.core.checkpoint_id_types import (
    CheckpointContainerId,
    CheckpointObjectId,
)
from ml_flashpoint.core.mlf_logging import get_logger
from ml_flashpoint.core.utils import log_execution_time
from ml_flashpoint.replication.transfer_service import transfer_service_ext

_LOGGER = get_logger(__name__)
_PROCESS_LOCAL_STRATEGY_INSTANCE = None
_PROCESS_LOCAL_REPLICATION_MANAGER_INSTANCE = None


class ReplicationStrategy:
    """Defines the interface for implementing replication strategies.

    Subclasses should implement get_destination_addresses to specify how
    replication targets are chosen based on the network topology.
    """

    def __init__(self, replication_service_addresses: List[str]):
        """Initializes the replication strategy.

        Args:
            replication_service_addresses: A list of all service addresses in the cluster indexed by global rank.
        """
        self._replication_service_addresses = replication_service_addresses

    @abc.abstractmethod
    def get_destination_addresses(self, global_rank: int) -> List[str]:
        """Determines the destination addresses for a given rank.

        Args:
            global_rank: The global rank of the process to get destination addresses for.

        Returns:
            A list of destination addresses for the given rank, in "<IP>:<port>" format.
        """
        raise NotImplementedError()

    def get_replication_service_addresses(self) -> List[str]:
        """Returns the list of all replication service addresses indexed by global rank.

        Returns:
            The list of all replication service addresses.
        """
        return self._replication_service_addresses


def _get_process_local_strategy_instance(replication_service_addresses, processes_per_node):
    global _PROCESS_LOCAL_STRATEGY_INSTANCE
    if _PROCESS_LOCAL_STRATEGY_INSTANCE is None:
        _PROCESS_LOCAL_STRATEGY_INSTANCE = PairwiseReplicationStrategy(
            replication_service_addresses, processes_per_node
        )
    return _PROCESS_LOCAL_STRATEGY_INSTANCE


class PairwiseReplicationStrategy(ReplicationStrategy):
    """Implements a replication strategy where ranks are paired for replication.

    This strategy pairs ranks one-to-one (e.g., 0<>1, 2<>3), so that each rank
    replicates its data to its peer. It requires an even number of total
    ranks to ensure that every rank is paired.
    """

    def __init__(self, replication_service_addresses: List[str], processes_per_node: int) -> None:
        """Initializes the pairwise replication strategy.

        Args:
            replication_service_addresses: A list of all service addresses in the cluster.
            processes_per_node: The number of processes per node.

        Raises:
            ValueError: If replication_service_addresses is empty, or if the total number of ranks is odd.
        """
        super().__init__(replication_service_addresses)
        if not self._replication_service_addresses:
            raise ValueError("The replication_service_addresses cannot be empty.")

        self._processes_per_node = processes_per_node
        self._world_size = len(self._replication_service_addresses)

        if self._world_size % self._processes_per_node != 0:
            raise ValueError(
                f"World size ({self._world_size}) must be divisible by processes per node ({self._processes_per_node})."
            )

        self._num_nodes = utils.get_num_of_nodes()
        if self._num_nodes % 2 != 0:
            raise ValueError(f"The total number of nodes ({self._num_nodes}) must be even.")

        if self._num_nodes != self._world_size // self._processes_per_node:
            raise ValueError(
                f"The total number of nodes ({self._num_nodes}) must be equal to the world size "
                "({self._world_size}) divided by the processes per node ({self._processes_per_node})."
            )

    def __reduce__(self):
        """Custom pickling to return the process-local singleton instance."""
        return (
            _get_process_local_strategy_instance,
            (self._replication_service_addresses, self._processes_per_node),
        )

    @override
    def get_destination_addresses(self, global_rank: int) -> List[str]:
        if not 0 <= global_rank < self._world_size:
            raise ValueError(f"global_rank {global_rank} is out of valid range [0, {self._world_size - 1}].")

        # 1. Convert the global rank to its corresponding node index and local rank.
        node_index = global_rank // self._processes_per_node
        local_rank = global_rank % self._processes_per_node

        # 2. Find the destination node index using the pairwise strategy.
        destination_node_index = node_index ^ 1

        # 3. Calculate the destination global rank (same local rank on peer node).
        destination_global_rank = destination_node_index * self._processes_per_node + local_rank

        # 4. Look up the destination rank's address.
        destination_address = self._replication_service_addresses[destination_global_rank]

        return [destination_address]


class ReplicationRetryConfig:
    """Configuration for controlling retry behavior in replication tasks."""

    def __init__(self, max_retries: int = 3, timeout_seconds: float = 30.0):
        self._max_retries = max_retries
        self._timeout_seconds = timeout_seconds

    def get_timeout(self) -> float:
        return self._timeout_seconds

    def get_retries(self) -> int:
        return self._max_retries


def get_default_retry_config() -> ReplicationRetryConfig:
    """Returns a default ReplicationRetryConfig instance.

    Returns:
        A default ReplicationRetryConfig instance.
    """
    return ReplicationRetryConfig()


def _get_process_local_replication_manager_instance():
    """Returns the process-local singleton instance of ReplicationManager.

    If one does not exist, it is created. This ensures we reuse the same manager
    (and its underlying C++ TransferService) in the worker process.
    """
    global _PROCESS_LOCAL_REPLICATION_MANAGER_INSTANCE
    if _PROCESS_LOCAL_REPLICATION_MANAGER_INSTANCE is None:
        _PROCESS_LOCAL_REPLICATION_MANAGER_INSTANCE = ReplicationManager()
    return _PROCESS_LOCAL_REPLICATION_MANAGER_INSTANCE


class ReplicationManager:
    """Manages asynchronous data replication using a C++ TransferService.

    The ReplicationManager orchestrates non-blocking replication tasks for
    checkpoint objects across different nodes. It leverages a C++-implemented
    TransferService for efficient data transfer and integrates with
    ReplicationStrategy to determine replication destinations.

    This class is responsible for:
    - Initializing and managing the lifecycle of the TransferService.
    - Applying a chosen ReplicationStrategy to identify target nodes for data.
    - Spawning asynchronous replication tasks and managing their completion
      through callbacks.
    - Ensuring proper cleanup and error handling for replication operations.
    """

    def __reduce__(self):
        """Custom pickling to return the process-local singleton instance.

        This allows the ReplicationManager to be passed to worker processes (via
        AsyncCallsQueue(persistent=True)) without trying to pickle the
        underlying C++ TransferService (which is not picklable) and instead
        using/initializing a local instance in the worker.
        """
        return (_get_process_local_replication_manager_instance, ())

    @log_execution_time(_LOGGER, "ReplicationManager.initialize")
    def initialize(
        self,
        checkpoint_object_manager: CheckpointObjectManager,
        listen_port: int = 0,
        retry_config: ReplicationRetryConfig = get_default_retry_config(),
        replication_transfer_service: Optional[transfer_service_ext.TransferService] = None,
        repl_strategy: Optional[ReplicationStrategy] = None,
    ) -> None:
        """Initializes the ReplicationManager and underlying services.

        This method sets up the TransferService for network communication,
        gathers network addresses from all ranks, and establishes a replication
        strategy. It must be called on all ranks simultaneously.

        Args:
            checkpoint_object_manager: The manager for checkpoint objects.
            listen_port: The port to listen on for incoming connections.
            retry_config: Configuration for replication retries.
            replication_transfer_service: An optional pre-configured
                TransferService instance. If not provided, a new one will be
                created.
            repl_strategy: An optional replication strategy. If not provided, a
                default PairwiseReplicationStrategy will be used.
        """
        self._checkpoint_object_manager = checkpoint_object_manager
        self._listen_port = listen_port
        self._retry_config = retry_config
        self._repl_strategy = repl_strategy
        self._transfer_service = replication_transfer_service
        self._initialize_address()

        if self._transfer_service is None:
            _LOGGER.info("No TransferService provided, initializing a new one...")
            self._transfer_service = transfer_service_ext.TransferService()
            bound_listen_port = self._transfer_service.initialize(
                listen_port,
                global_rank=dist.get_rank(),
            )
            if bound_listen_port <= 0:
                _LOGGER.error("Failed to initialize C++ TransferService")
                raise RuntimeError("Failed to initialize TransferService")
            self._listen_port = bound_listen_port
            _LOGGER.info("C++ TransferService initialized successfully.")
        else:
            _LOGGER.info("Using pre-configured TransferService.")

        if self._repl_strategy is None:
            self._gather_replication_service_addresses()
            _LOGGER.warning("No ReplicationStrategy provided.")
            repl_strategy = PairwiseReplicationStrategy(
                self._replication_service_addr_global_view, torch.cuda.device_count()
            )
            self._repl_strategy = repl_strategy
            _LOGGER.info("Using default ReplicationStrategy.")
        else:
            _LOGGER.info("Using pre-configured ReplicationStrategy.")
            if hasattr(self._repl_strategy, "get_replication_service_addresses"):
                self._replication_service_addr_global_view = self._repl_strategy.get_replication_service_addresses()

    @log_execution_time(_LOGGER, "ReplicationManager.async_replicate")
    def async_replicate(self, buffer_io: BufferIO) -> list[concurrent.futures.Future]:
        """Asynchronously replicates a checkpoint object to peer nodes.

        This non-blocking method initiates replication tasks for the given
        buffer and returns a list of futures that can be used to track their
        completion. The buffer is automatically closed once all replication
        tasks for it have finished.

        Args:
            buffer_io: The buffer to replicate.

        Returns:
            A list of concurrent.futures.Future objects, one for each
            replication task.
        """

        if self._repl_strategy is None:
            _LOGGER.error("No replication strategy set. Cannot replicate.")
            return []

        if self._transfer_service is None:
            _LOGGER.error("TransferService is not initialized. Cannot replicate.")
            return []

        obj_id = str(buffer_io.buffer_obj.get_id())
        _LOGGER.info("Replicating object '%s'.", obj_id)

        # Get destination addresses
        destination_addresses = self._repl_strategy.get_destination_addresses(dist.get_rank())
        if not destination_addresses:
            _LOGGER.warning("No destinations found for '%s'. Replication skipped.", obj_id)
            # If no destinations, we can close the buffer immediately
            # TODO: Change all the f-string to placeholder in python logging.
            _LOGGER.info("No replication needed for '%s', closing buffer.", obj_id)
            try:
                self._checkpoint_object_manager.close_buffer(buffer_io)
            except Exception:
                _LOGGER.exception("Failed to close buffer '%s'", obj_id)
            return []
        futures = []
        start_repl_time = time.perf_counter()
        for destination_address in destination_addresses:
            future = self._transfer_service.async_put(
                buffer_io.buffer_obj.get_data_ptr(),
                buffer_io.buffer_obj.get_capacity(),
                destination_address,
                obj_id,
            )
            futures.append(future)

        self._add_aggregate_callback(futures, lambda f: self._final_replication_callback(buffer_io, f, start_repl_time))
        _LOGGER.info("Aggregate callback has been attached. Waiting for completion...")
        return futures

    @staticmethod
    @log_execution_time(_LOGGER, "ReplicationManager._add_aggregate_callback")
    def _add_aggregate_callback(
        futures: List[concurrent.futures.Future],
        final_callback: Callable[[List[concurrent.futures.Future]], None],
    ) -> None:
        """Attaches a callback that triggers after all futures are complete.

        This method uses a thread-safe counter to track the completion of all
        futures in the provided list. The final_callback is invoked only once,
        after the last future has completed.

        Args:
            futures: A list of concurrent.futures.Future objects to track.
            final_callback: The function to call when all futures are complete.
                It will be passed the original list of futures.
        """
        if not futures:
            final_callback([])
            return

        lock = threading.Lock()
        remaining = len(futures)

        def done_callback(future: concurrent.futures.Future) -> None:
            """Decrements the counter and calls the final callback when it hits zero.

            Args:
                future: The future that just completed.
            """
            nonlocal remaining
            with lock:
                remaining -= 1
                if remaining == 0:
                    _LOGGER.info("All futures have completed.")
                    final_callback(futures)
                else:
                    _LOGGER.debug("%s futures remaining.", remaining)

        for future in futures:
            future.add_done_callback(done_callback)

    @log_execution_time(_LOGGER, "ReplicationManager._final_replication_callback")
    def _final_replication_callback(
        self, buffer_io: BufferIO, completed_futures: List[concurrent.futures.Future], repl_start_time: float
    ) -> None:
        """Handles the completion of all replication tasks for a buffer.

        This callback is triggered once all futures for a BufferIO's
        replication are complete. It logs any errors and closes the buffer.

        Args:
            buffer_io: The BufferIO object that was replicated.
            completed_futures: A list of futures for the completed replication
                tasks.
            repl_start_time: The time replication was initiated, as obtained by time.perf_counter(). Used for logging.
        """
        _LOGGER.info("Executing the final replication callback.")
        errors = []
        for f in completed_futures:
            exc = f.exception()
            if exc:
                errors.append(exc)
                _LOGGER.error(
                    "Buffer object '%s' replication failed with exception: '%s'",
                    buffer_io.buffer_obj.get_id(),
                    exc,
                )
            else:
                result = f.result()
                if not result.success:
                    error_msg = f"Buffer object {buffer_io.buffer_obj.get_id()} "
                    f"replication failed: {result.error_message}"
                    errors.append(RuntimeError(error_msg))
                    _LOGGER.error(error_msg)

        _LOGGER.info(
            "Replication of '%s' took %.4fs",
            Path(buffer_io.buffer_obj.get_id()).name,
            time.perf_counter() - repl_start_time,
        )
        # Close buffer when all the replication tasks for the buffer object are done.
        with log_execution_time(_LOGGER, "_final_replication_callback__close_buffer", level=logging.DEBUG):
            self._checkpoint_object_manager.close_buffer(buffer_io)
        if buffer_io.buffer_obj:
            _LOGGER.info("Buffer '%s' closed.", buffer_io.buffer_obj.get_id())
        if errors:
            _LOGGER.error("%d replications failed: '%s'", len(errors), errors)
        else:
            _LOGGER.info("All replications completed successfully.")

    def _async_retrieve(
        self,
        source_address: str,
        object_id_to_retrieve: CheckpointObjectId,
        retrieved_object_id: CheckpointObjectId,
    ) -> concurrent.futures.Future:
        """Initiates an asynchronous retrieval from a peer node.

        Args:
            source_address: The address of the peer to fetch from.
            object_id_to_retrieve: The ID of the object to fetch.
            retrieved_object_id: The ID to assign to the retrieved object.


        Returns:
            A concurrent.futures.Future that will contain the TransferResult.
        """
        if self._transfer_service is None:
            _LOGGER.error("TransferService is not initialized. Cannot retrieve.")
            # Return a future that is already failed
            future = concurrent.futures.Future()
            future.set_exception(RuntimeError("TransferService is not initialized."))
            return future

        _LOGGER.info(
            "Requesting retrieval of '%s' from '%s'",
            object_id_to_retrieve,
            source_address,
        )
        return self._transfer_service.async_get(str(object_id_to_retrieve), source_address, str(retrieved_object_id))

    @log_execution_time(_LOGGER, "ReplicationManager.sync_bulk_retrieve")
    def sync_bulk_retrieve(
        self,
        source_global_rank: int,
        object_ids_to_retrieve: List[CheckpointObjectId],
        container_ids_to_retrieve: List[CheckpointContainerId],
        retrieved_object_ids: List[CheckpointObjectId] | None = None,
        retrieved_container_ids: List[CheckpointContainerId] | None = None,
    ) -> bool:
        """Synchronously retrieves a collection of checkpoint objects.

        This blocking method fetches all specified checkpoint objects from a
        remote source. It waits until the entire bulk retrieval operation is
        complete before returning.

        Args:
            source_global_rank: The global rank of the source node.
            object_ids_to_retrieve: A list of specific checkpoint object IDs to
                retrieve.
            container_ids_to_retrieve: A list of checkpoint container IDs, from
                which all objects will be retrieved.
            retrieved_object_ids: A list of retrieved object ids, if None, use the original object ids.
            retrieved_container_ids: A list of retrieved container ids, if None, use the original container ids.

        Returns:
            `True` if the bulk retrieval was successful, `False` otherwise.
        """
        if self._retry_config is None:
            _LOGGER.error("ReplicationManager is not initialized. Cannot retrieve.")
            return False

        if retrieved_object_ids and len(retrieved_object_ids) != len(object_ids_to_retrieve):
            _LOGGER.error("Retrieved_object_ids must be empty or the same length as object_ids_to_retrieve.")
            return False

        if retrieved_container_ids and len(retrieved_container_ids) != len(container_ids_to_retrieve):
            _LOGGER.error("Retrieved_container_ids must be empty or the same length as container_ids_to_retrieve.")

        _LOGGER.info(
            "Starting sync bulk retrieve of %d objects from '%s'",
            len(object_ids_to_retrieve),
            source_global_rank,
        )

        if not (0 <= source_global_rank < len(self._replication_service_addr_global_view)):
            _LOGGER.error("Invalid source_global_rank: %d", source_global_rank)
            return False
        source_address = self._replication_service_addr_global_view[source_global_rank]

        futures = []
        effective_retrieved_ids = retrieved_object_ids if retrieved_object_ids else object_ids_to_retrieve
        for i, obj_id in enumerate(object_ids_to_retrieve):
            futures.append(self._async_retrieve(source_address, obj_id, effective_retrieved_ids[i]))

        # TODO: Handle container_ids_to_retrieve

        done, not_done = concurrent.futures.wait(futures, timeout=self._retry_config.get_timeout())

        if not_done:
            _LOGGER.error("Timed out waiting for %d retrievals.", len(not_done))
            return False

        all_success = True
        for f in done:
            try:
                result = f.result()
                if not result.success:
                    _LOGGER.error("Retrieval failed: %s", result.error_message)
                    all_success = False
            except Exception:
                _LOGGER.exception("Retrieval failed with exception")
                all_success = False

        return all_success

    @log_execution_time(_LOGGER, "ReplicationManager.shutdown")
    def shutdown(self):
        """Shuts down the ReplicationManager and its TransferService."""
        _LOGGER.info("Shutting down ReplicationManager and TransferService...")
        if self._transfer_service:
            self._transfer_service.shutdown()
            _LOGGER.info("TransferService shut down.")

    # TODO: Use the ip address return from c++ transfer service to avoid duplication.
    def _initialize_address(self):
        """Initializes the local IP address for the replication service."""
        try:
            hostname = socket.gethostname()
            self._hostname = socket.gethostbyname(hostname)
        except Exception:
            _LOGGER.exception("Failed to initialize replication address")
            self._hostname = "Error"

    @log_execution_time(_LOGGER, "ReplicationManager._gather_replication_service_addresses")
    def _gather_replication_service_addresses(self):
        """Gathers and distributes replication service addresses across all ranks."""
        local_address = f"{self._hostname}:{self._listen_port}"
        _LOGGER.info("Local address: '%s'", local_address)
        world_size = dist.get_world_size()
        # Ensure the list is clean and properly sized for object gathering
        self._replication_service_addr_global_view = [None] * world_size
        try:
            _LOGGER.info(
                "Global rank '%s' gathering replication service addresses. Local address: '%s'",
                dist.get_rank(),
                local_address,
            )
            dist.all_gather_object(self._replication_service_addr_global_view, local_address)
            _LOGGER.info(
                "Global rank '%s' successfully gathered addresses: '%s'",
                dist.get_rank(),
                self._replication_service_addr_global_view,
            )
        except Exception as e:
            _LOGGER.exception(
                "Global rank '%s' failed to gather replication service addresses",
                dist.get_rank(),
            )
            # Clear the view on failure to indicate an incomplete or erroneous state
            self._replication_service_addr_global_view = []
            raise RuntimeError(f"Failed to gather replication service addresses from all ranks. Error: {e}")
