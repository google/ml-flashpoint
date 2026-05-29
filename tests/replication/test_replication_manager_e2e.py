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
import threading
import time

import pytest

from ml_flashpoint.checkpoint_object_manager.buffer_io import BufferIO
from ml_flashpoint.checkpoint_object_manager.buffer_object import buffer_object_ext
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.replication.replication_manager import PairwiseReplicationStrategy, ReplicationManager
from ml_flashpoint.replication.transfer_service import transfer_service_ext


def run_service(service, port_container, rank, repl_shm_name=""):
    """Target function for running a TransferService in a thread."""
    port = service.initialize(0, global_rank=rank, repl_shm_name=repl_shm_name)
    port_container.append(port)


@pytest.fixture
def services():
    """Fixture to set up and tear down sender and receiver TransferServices."""
    sender_service = transfer_service_ext.TransferService()
    receiver_service = transfer_service_ext.TransferService()

    sender_port_container = []
    receiver_port_container = []

    # Start receiver service to get its port
    receiver_thread = threading.Thread(target=run_service, args=(receiver_service, receiver_port_container, 1))
    receiver_thread.daemon = True
    receiver_thread.start()

    # Start sender service
    sender_thread = threading.Thread(target=run_service, args=(sender_service, sender_port_container, 0))
    sender_thread.daemon = True
    sender_thread.start()

    # Wait for the services to initialize and get their ports
    start_time = time.time()
    while (not sender_port_container or not receiver_port_container) and time.time() - start_time < 5:
        time.sleep(0.1)

    assert sender_port_container, "Sender service failed to start"
    assert receiver_port_container, "Receiver service failed to start"

    sender_port = sender_port_container[0]
    receiver_port = receiver_port_container[0]

    sender_addr = f"127.0.0.1:{sender_port}"
    receiver_addr = f"127.0.0.1:{receiver_port}"

    yield sender_service, receiver_service, sender_addr, receiver_addr

    # Teardown
    sender_service.shutdown()
    receiver_service.shutdown()
    sender_thread.join(timeout=5)
    receiver_thread.join(timeout=5)


@pytest.mark.e2e
def test_async_replicate_end_to_end(tmp_path, services, mocker):
    """
    An end-to-end test for async_replicate using real TransferServices.
    """
    # Given
    sender_service, _, sender_addr, receiver_addr = services

    # Mock torch.distributed to simulate a 2-rank environment where rank 0 is the sender
    mocker.patch("torch.distributed.get_rank", return_value=0)
    mocker.patch("torch.distributed.get_world_size", return_value=2)
    mocker.patch("torch.cuda.device_count", return_value=1)
    mocker.patch("ml_flashpoint.core.utils.get_num_of_nodes", return_value=2)

    # Create a real BufferObject for the sender
    # The replicated file will be created with this same path on the receiver side
    obj_id = str(tmp_path / "test_buffer")
    capacity = 4096 * 2
    buffer_object = buffer_object_ext.BufferObject(obj_id, capacity, overwrite=True)

    # Write some data to it
    original_data = os.urandom(4096)
    buffer_io = BufferIO(buffer_object)
    buffer_io.write(original_data)

    # Setup ReplicationManager for the sender
    ckpt_obj_manager = CheckpointObjectManager()
    manager = ReplicationManager()
    # In a 2-node, and each node has 1 rank setup, rank 0 replicates to rank 1
    strategy = PairwiseReplicationStrategy(
        replication_service_addresses=[sender_addr, receiver_addr], processes_per_node=1
    )

    # Initialize the manager with the real sender service and the strategy
    manager.initialize(
        checkpoint_object_manager=ckpt_obj_manager, replication_transfer_service=sender_service, repl_strategy=strategy
    )

    # Replicate the buffer
    futures = manager.async_replicate(buffer_io)

    # Then
    assert len(futures) == 1
    result = futures[0].result()
    assert result.success

    # Verify the replicated data on the receiver side
    replicated_bo = buffer_object_ext.BufferObject(obj_id)
    replicated_buffer_io = BufferIO(replicated_bo)
    replicated_data = replicated_buffer_io.read()
    assert replicated_data == original_data
    ckpt_obj_manager.close_buffer(replicated_buffer_io)


@pytest.mark.e2e
def test_sync_bulk_retrieve_end_to_end(tmp_path, services, mocker):
    """
    An end-to-end test for sync_bulk_retrieve using real TransferServices.
    """
    # Given
    _, receiver_service, sender_addr, _ = services

    # Mock torch.distributed to simulate a 2-rank environment where rank 1 is the receiver
    mocker.patch("torch.distributed.get_rank", return_value=1)
    mocker.patch("torch.distributed.get_world_size", return_value=2)
    mocker.patch("torch.cuda.device_count", return_value=1)
    mocker.patch("ml_flashpoint.core.utils.get_num_of_nodes", return_value=2)

    # Create multiple BufferObjects on the sender side
    num_objects = 5
    original_data_map = {}
    obj_ids = []
    retrieved_obj_ids = []
    buffer_ios = []  # Keep BufferIO objects in scope
    for i in range(num_objects):
        obj_id = str(tmp_path / f"test_buffer_{i}")
        capacity = 4096 * 2
        buffer_object = buffer_object_ext.BufferObject(obj_id, capacity, overwrite=True)
        original_data = os.urandom(2048)
        buffer_io = BufferIO(buffer_object)
        buffer_io.write(original_data)
        buffer_io.close()
        original_data_map[obj_id] = original_data
        obj_ids.append(obj_id)
        retrieved_obj_ids.append(obj_id + "_retrieved")
        buffer_ios.append(buffer_io)

    # Setup ReplicationManager for the receiver
    ckpt_obj_manager = CheckpointObjectManager()
    manager = ReplicationManager()
    strategy = PairwiseReplicationStrategy(
        replication_service_addresses=[sender_addr, "127.0.0.1:0"], processes_per_node=1
    )
    manager.initialize(
        checkpoint_object_manager=ckpt_obj_manager,
        replication_transfer_service=receiver_service,
        repl_strategy=strategy,
    )

    # When
    success = manager.sync_bulk_retrieve(
        source_global_rank=0,
        object_ids_to_retrieve=obj_ids,
        container_ids_to_retrieve=[],
        retrieved_object_ids=retrieved_obj_ids,
        retrieved_container_ids=[],
    )

    # Then
    assert success

    # Verify the retrieved data
    for obj_id, retrieved_obj_id in zip(obj_ids, retrieved_obj_ids):
        retrieved_bo = buffer_object_ext.BufferObject(retrieved_obj_id)
        retrieved_buffer_io = BufferIO(retrieved_bo)
        retrieved_data = retrieved_buffer_io.read()
        assert retrieved_data == original_data_map[obj_id]
        ckpt_obj_manager.close_buffer(retrieved_buffer_io)


@pytest.mark.e2e
def test_sync_bulk_retrieve_with_buffer_pool_and_resize(tmp_path, services, mocker):
    """
    An end-to-end test for sync_bulk_retrieve using BufferPool and triggering resize.
    """
    # Given
    # We ignore receiver_service from fixture because we need to create one with pool attached!
    sender_service, _, sender_addr, _ = services

    # Mock torch.distributed
    mocker.patch("torch.distributed.get_rank", return_value=1)
    mocker.patch("torch.distributed.get_world_size", return_value=2)
    mocker.patch("torch.cuda.device_count", return_value=1)
    mocker.patch("ml_flashpoint.core.utils.get_num_of_nodes", return_value=2)

    # METADATA_SIZE is 4096. We need buffer larger than that.
    metadata_size = 4096
    
    # Create a BufferObject on the sender side with size metadata_size + 2048
    obj_id = str(tmp_path / "test_buffer_large")
    capacity = metadata_size + 2048
    sender_bo = buffer_object_ext.BufferObject(obj_id, capacity, overwrite=True)
    original_data = os.urandom(2048)
    
    buffer_io = BufferIO(sender_bo)
    buffer_io.write(original_data)
    buffer_io.close()

    # Setup CheckpointObjectManager with BufferPoolConfig on receiver
    # Set default buffer_size small (e.g. metadata_size + 1024) to trigger resize!
    pool_dir = tmp_path / "pool_dir"
    os.makedirs(pool_dir)
    from ml_flashpoint.core.buffer_pool import BufferPoolConfig
    pool_config = BufferPoolConfig(
        pool_dir_path=str(pool_dir),
        rank=1,
        num_buffers=2,
        buffer_size=metadata_size + 1024, # Smaller than metadata_size + 2048!
    )
    ckpt_obj_manager = CheckpointObjectManager(repl_pool_config=pool_config)
    
    # Create and start a NEW receiver TransferService with pool attached!
    new_receiver_service = transfer_service_ext.TransferService()
    repl_shm_name = ckpt_obj_manager.replication_pool_shm_name
    receiver_port_container = []
    
    import threading
    import time
    
    receiver_thread = threading.Thread(
        target=run_service, 
        args=(new_receiver_service, receiver_port_container, 1, repl_shm_name)
    )
    receiver_thread.daemon = True
    receiver_thread.start()
    
    start_time = time.time()
    while not receiver_port_container and time.time() - start_time < 5:
        time.sleep(0.1)
    assert receiver_port_container, "New receiver service failed to start"
    
    manager = ReplicationManager()
    strategy = PairwiseReplicationStrategy(
        replication_service_addresses=[sender_addr, "127.0.0.1:0"], processes_per_node=1
    )
    manager.initialize(
        checkpoint_object_manager=ckpt_obj_manager,
        replication_transfer_service=new_receiver_service,
        repl_strategy=strategy,
    )

    retrieved_obj_id = obj_id + "_retrieved"

    try:
        # When
        success = manager.sync_bulk_retrieve(
            source_global_rank=0,
            object_ids_to_retrieve=[obj_id],
            container_ids_to_retrieve=[],
            retrieved_object_ids=[retrieved_obj_id],
            retrieved_container_ids=[],
        )
    
        # Then
        assert success
    
        # Verify that retrieved_obj_id is a symlink!
        assert os.path.islink(retrieved_obj_id)
        
        # Verify the retrieved data
        retrieved_bo = buffer_object_ext.BufferObject(retrieved_obj_id)
        retrieved_buffer_io = BufferIO(retrieved_bo)
        retrieved_data = retrieved_buffer_io.read()
        assert retrieved_data == original_data
        ckpt_obj_manager.close_buffer(retrieved_buffer_io)
        
    finally:
        new_receiver_service.shutdown()
        receiver_thread.join(timeout=5)
