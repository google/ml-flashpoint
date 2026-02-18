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

from concurrent.futures import Future

import pytest

from ml_flashpoint.replication.replication_manager import ReplicationManager, ReplicationRetryConfig


@pytest.fixture
def replication_manager(mocker):
    """Provides a ReplicationManager instance with mocked dependencies."""
    manager = ReplicationManager()
    manager._transfer_service = mocker.MagicMock()
    manager._repl_strategy = mocker.MagicMock()
    manager._checkpoint_object_manager = mocker.MagicMock()
    manager._retry_config = ReplicationRetryConfig(timeout_seconds=1)
    return manager


def test_initialize_binds_port_success(replication_manager, mocker):
    """Tests that initialize correctly binds the port when TransferService succeeds."""
    # Given
    replication_manager._transfer_service = None
    mock_transfer_service_cls = mocker.patch(
        "ml_flashpoint.replication.transfer_service.transfer_service_ext.TransferService"
    )
    mock_transfer_service_instance = mock_transfer_service_cls.return_value
    mock_transfer_service_instance.initialize.return_value = 12345
    mock_checkpoint_manager = mocker.MagicMock()
    mocker.patch("torch.cuda.device_count", return_value=1)
    mocker.patch("ml_flashpoint.core.utils.get_num_of_nodes", return_value=2)
    mocker.patch("torch.distributed.get_rank", return_value=0)

    # Mock _gather_replication_service_addresses to avoid network calls and set necessary attribute
    def mock_gather():
        replication_manager._replication_service_addr_global_view = ["addr1", "addr2"]

    replication_manager._gather_replication_service_addresses = mocker.MagicMock(side_effect=mock_gather)

    # When
    replication_manager.initialize(mock_checkpoint_manager, listen_port=0)

    # Then
    assert replication_manager._listen_port == 12345
    mock_transfer_service_instance.initialize.assert_called_once_with(0, global_rank=0)


def test_initialize_binds_port_failure(replication_manager, mocker):
    """Tests that initialize raises RuntimeError when TransferService fails to bind port."""
    # Given
    replication_manager._transfer_service = None
    mock_transfer_service_cls = mocker.patch(
        "ml_flashpoint.replication.transfer_service.transfer_service_ext.TransferService"
    )
    mock_transfer_service_instance = mock_transfer_service_cls.return_value
    mock_transfer_service_instance.initialize.return_value = 0  # Simulate failure
    mock_checkpoint_manager = mocker.MagicMock()
    # Mock _gather_replication_service_addresses
    replication_manager._gather_replication_service_addresses = mocker.MagicMock()
    mocker.patch("torch.distributed.get_rank", return_value=20)

    # When/Then
    with pytest.raises(RuntimeError, match="Failed to initialize TransferService"):
        replication_manager.initialize(mock_checkpoint_manager)


def test_async_replicate_no_strategy(replication_manager, mocker):
    """Tests that async_replicate returns an empty list if no strategy is set."""
    # Given
    replication_manager._repl_strategy = None
    buffer_object = mocker.MagicMock()

    # When
    result = replication_manager.async_replicate(buffer_object)

    # Then
    assert result == []


def test_async_replicate_no_transfer_service(replication_manager, mocker):
    """Tests that async_replicate returns an empty list if the transfer service is not set."""
    # Given
    replication_manager._transfer_service = None
    buffer_object = mocker.MagicMock()

    # When
    result = replication_manager.async_replicate(buffer_object)

    # Then
    assert result == []


def test_async_replicate_no_destination(replication_manager, mocker):
    """Tests that async_replicate returns an empty list and closes the buffer if no destination is found."""
    # Given
    mocker.patch("torch.distributed.get_rank", return_value=0)
    replication_manager._repl_strategy.get_destination_addresses.return_value = []
    buffer_object = mocker.MagicMock()
    buffer_object.get_id.return_value = "test_obj_id"
    buffer_io = mocker.MagicMock(buffer_obj=buffer_object)

    # When
    result = replication_manager.async_replicate(buffer_io)

    # Then
    assert result == []
    replication_manager._checkpoint_object_manager.close_buffer.assert_called_once_with(buffer_io)


def test_async_replicate_success(replication_manager, mocker):
    """Tests that async_replicate calls async_put with the correct arguments for each destination."""
    # Given
    mocker.patch("torch.distributed.get_rank", return_value=0)
    replication_manager._repl_strategy.get_destination_addresses.return_value = [
        "dest1",
        "dest2",
    ]
    buffer_object = mocker.MagicMock()
    buffer_object.get_id.return_value = "test_obj_id"
    buffer_object.get_data_ptr.return_value = 1234
    buffer_object.get_capacity.return_value = 1024
    buffer_io = mocker.MagicMock(buffer_obj=buffer_object)

    future1 = Future()
    future2 = Future()
    replication_manager._transfer_service.async_put.side_effect = [future1, future2]
    replication_manager._add_aggregate_callback = mocker.MagicMock()

    # When
    result_futures = replication_manager.async_replicate(buffer_io)

    # Then
    assert result_futures == [future1, future2]
    replication_manager._transfer_service.async_put.assert_any_call(1234, 1024, "dest1", "test_obj_id")
    replication_manager._transfer_service.async_put.assert_any_call(1234, 1024, "dest2", "test_obj_id")

    # Verify _add_aggregate_callback is called with the correct arguments
    replication_manager._add_aggregate_callback.assert_called_once()
    call_args, _ = replication_manager._add_aggregate_callback.call_args
    assert call_args[0] == [future1, future2]
    assert callable(call_args[1])


def test_final_replication_callback_success(replication_manager, mocker):
    """Tests that _final_replication_callback closes buffers on success."""
    # Given
    mocker.patch("torch.distributed.get_rank", return_value=0)
    mock_buffer_io = mocker.MagicMock()
    mock_buffer_io.buffer_obj.get_id.return_value = "obj_id_1"

    future1 = Future()
    future1.set_result(mocker.MagicMock(success=True))
    future2 = Future()
    future2.set_result(mocker.MagicMock(success=True))

    completed_futures = [future1, future2]
    mock_logger = mocker.patch("ml_flashpoint.replication.replication_manager._LOGGER")

    # When
    replication_manager._final_replication_callback(mock_buffer_io, completed_futures, 123.45)

    # Then
    replication_manager._checkpoint_object_manager.close_buffer.assert_called_once_with(mock_buffer_io)
    mock_logger.error.assert_not_called()


def test_final_replication_callback_failure(replication_manager, mocker):
    """Tests that _final_replication_callback logs errors and closes buffers on failure."""
    # Given
    mocker.patch("torch.distributed.get_rank", return_value=0)
    mock_buffer_io = mocker.MagicMock()
    mock_buffer_io.buffer_obj.get_id.return_value = "obj_id_1"

    future1 = Future()
    future1.set_result(mocker.MagicMock(success=True))
    future2 = Future()
    future2.set_exception(RuntimeError("Replication failed"))

    completed_futures = [future1, future2]
    mock_logger = mocker.patch("ml_flashpoint.replication.replication_manager._LOGGER")

    # When
    replication_manager._final_replication_callback(mock_buffer_io, completed_futures, 123.45)

    # Then
    replication_manager._checkpoint_object_manager.close_buffer.assert_called_once_with(mock_buffer_io)
    mock_logger.error.assert_any_call(
        "Buffer object '%s' replication failed with exception: '%s'",
        "obj_id_1",
        mocker.ANY,
    )
    mock_logger.error.assert_any_call("%d replications failed: '%s'", 1, mocker.ANY)


def test_async_retrieve_success(replication_manager):
    """Tests that _async_retrieve calls async_get with the correct arguments."""
    # Given
    obj_id = "test_obj_id"
    source_address = "test_address"
    replication_manager._replication_service_addr_global_view = [source_address]

    # When
    replication_manager._async_retrieve(source_address, obj_id, obj_id)

    # Then
    replication_manager._transfer_service.async_get.assert_called_once_with(obj_id, source_address, obj_id)


def test_sync_bulk_retrieve_success(replication_manager, mocker):
    """Tests that sync_bulk_retrieve returns True on success."""
    # Given
    obj_ids = ["obj1", "obj2"]
    container_ids = []
    replication_manager._replication_service_addr_global_view = ["addr1"]

    future1 = Future()
    future1.set_result(mocker.MagicMock(success=True))
    future2 = Future()
    future2.set_result(mocker.MagicMock(success=True))
    mocker.patch.object(replication_manager, "_async_retrieve", side_effect=[future1, future2])

    # When
    result = replication_manager.sync_bulk_retrieve(0, obj_ids, container_ids)

    # Then
    assert result is True
    assert replication_manager._async_retrieve.call_count == 2


def test_sync_bulk_retrieve_failure(replication_manager, mocker):
    """Tests that sync_bulk_retrieve returns False on failure."""
    # Given
    obj_ids = ["obj1", "obj2"]
    container_ids = []

    future1 = Future()
    future1.set_result(mocker.MagicMock(success=True))
    future2 = Future()
    future2.set_result(mocker.MagicMock(success=False, error_message="Failed"))
    replication_manager._replication_service_addr_global_view = ["addr1"]
    mocker.patch.object(replication_manager, "_async_retrieve", side_effect=[future1, future2])

    # When
    result = replication_manager.sync_bulk_retrieve(0, obj_ids, container_ids)

    # Then
    assert result is False


def test_sync_bulk_retrieve_exception(replication_manager, mocker):
    """Tests that sync_bulk_retrieve returns False on exception."""
    # Given
    obj_ids = ["obj1", "obj2"]
    container_ids = ["cont1", "cont2"]
    replication_manager._replication_service_addr_global_view = ["addr1"]

    future1 = Future()
    future1.set_result(mocker.MagicMock(success=True))
    future2 = Future()
    future2.set_exception(RuntimeError("Failed"))
    mocker.patch.object(replication_manager, "_async_retrieve", side_effect=[future1, future2])

    # When
    result = replication_manager.sync_bulk_retrieve(0, obj_ids, container_ids)

    # Then
    assert result is False


def test_sync_bulk_retrieve_timeout(replication_manager, mocker):
    """Tests that sync_bulk_retrieve returns False on timeout."""
    # Given
    obj_ids = ["obj1"]
    container_ids = []
    replication_manager._replication_service_addr_global_view = ["addr1"]

    future = Future()  # Never gets a result
    mocker.patch.object(replication_manager, "_async_retrieve", return_value=future)

    # When
    result = replication_manager.sync_bulk_retrieve(0, obj_ids, container_ids)

    # Then
    assert result is False


def test_sync_bulk_retrieve_empty_list(replication_manager):
    """Tests that sync_bulk_retrieve returns True for an empty list."""
    # When
    replication_manager._replication_service_addr_global_view = ["addr1"]
    result = replication_manager.sync_bulk_retrieve(0, [], [])

    # Then
    assert result is True


def test_sync_bulk_retrieve_mismatched_retrieved_list(replication_manager):
    """Tests that sync_bulk_retrieve returns False for mismatched retrieved list."""
    # When
    replication_manager._replication_service_addr_global_view = ["addr1"]
    result = replication_manager.sync_bulk_retrieve(0, ["obj1", "obj2"], [], ["obj1"], [])

    # Then
    assert result is False


def test_sync_bulk_retrieve_invalid_rank(replication_manager):
    """Tests that sync_bulk_retrieve returns False when source_global_rank is invalid."""
    # Given
    obj_ids = ["obj1"]
    container_ids = []
    invalid_rank = 2
    replication_manager._replication_service_addr_global_view = ["addr1", "addr2"]

    # When
    result = replication_manager.sync_bulk_retrieve(invalid_rank, obj_ids, container_ids)

    # Then
    assert result is False
