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

"""Tests for the TransferService extension."""

import logging
import pathlib
import socket

import numpy as np
import pytest

from ml_flashpoint.replication.transfer_service import transfer_service_ext

_LOGGER = logging.getLogger(__name__)


def _get_free_port() -> int:
    """Finds and returns an available TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def safe_remove_file(filepath: str) -> None:
    """Removes a file if it exists, logging the action.

    This function will not raise an error if the file does not exist.

    Args:
        filepath: The string path to the file to be removed.
    """
    path_obj = pathlib.Path(filepath)
    try:
        path_obj.unlink(missing_ok=True)
        if not path_obj.exists():
            _LOGGER.info("Successfully removed file: %s", path_obj)
        else:
            # This case might occur if permissions are denied.
            _LOGGER.warning("File could not be removed: %s", path_obj)

    except IsADirectoryError:
        _LOGGER.exception("Path is a directory, not a file: %s", path_obj)
        # Re-raise the exception because this indicates a usage error.
        raise


@pytest.fixture(name="transfer_services")
def fixture_transfer_services() -> tuple[
    transfer_service_ext.TransferService,
    transfer_service_ext.TransferService,
    str,
    str,
]:
    """Initializes and yields two TransferService instances for testing.

    This fixture handles the lifecycle of two services, ensuring they are
    properly initialized before a test runs and shut down afterward.

    Yields:
        A tuple containing:
            - The first TransferService instance.
            - The second TransferService instance.
            - The network address (host:port) of the first service.
            - The network address (host:port) of the second service.
    """
    service1 = transfer_service_ext.TransferService()
    service2 = transfer_service_ext.TransferService()

    port1 = _get_free_port()
    port2 = _get_free_port()
    addr1 = f"127.0.0.1:{port1}"
    addr2 = f"127.0.0.1:{port2}"

    service1.initialize(listen_port=port1)
    service2.initialize(listen_port=port2)

    yield service1, service2, addr1, addr2

    # Teardown: This code runs after the test function completes.
    service1.shutdown()
    service2.shutdown()


def test_successful_transfer(
    transfer_services: tuple[
        transfer_service_ext.TransferService,
        transfer_service_ext.TransferService,
        str,
        str,
    ],
) -> None:
    """Verifies a successful async_put operation between two services.

    Args:
        transfer_services: A pytest fixture providing two initialized services.
    """
    service1, _, _, addr2 = transfer_services

    object_id = "test.object"
    data_to_send = np.arange(20, dtype=np.int64)

    # Call the async_put method to transfer data from service1 to service2.
    put_future = service1.async_put(
        data_to_send.ctypes.data,  # data_ptr
        data_to_send.nbytes,  # data_size
        addr2,  # dest_address
        object_id,  # dest_object_id
    )

    # Block until the C++ future completes and get the result.
    result = put_future.result(timeout=10)  # Add a 10 seconds timeout for safety.

    assert result.success is True, "The 'success' flag should be True on a successful transfer."
    assert not result.error_message, (
        f"The 'error_message' should be empty on a successful transfer, but was: '{result.error_message}'"
    )

    # Clean up
    safe_remove_file(object_id)


def test_async_put_non_blocking(
    transfer_services: tuple[
        transfer_service_ext.TransferService,
        transfer_service_ext.TransferService,
        str,
        str,
    ],
) -> None:
    """Verifies that async_put is non-blocking and the transfer happens in the background.

    Args:
        transfer_services: A pytest fixture providing two initialized services.
    """
    service1, _, _, addr2 = transfer_services

    object_id = "test_non_blocking_put.object"
    data_to_send = np.arange(20, dtype=np.int64)
    expected_content = data_to_send.tobytes()

    # Ensure the destination file does not exist before the test
    safe_remove_file(object_id)

    # Initiate the async_put operation
    put_future = service1.async_put(
        data_to_send.ctypes.data,  # data_ptr
        data_to_send.nbytes,  # data_size
        addr2,  # dest_address
        object_id,  # dest_object_id
    )

    # Block until the C++ future completes and get the result.
    result = put_future.result(timeout=10)

    assert result.success is True, "The 'success' flag should be True on a successful transfer."
    assert not result.error_message, (
        f"The 'error_message' should be empty on a successful transfer, but was: '{result.error_message}'"
    )

    # Now, verify the file exists and its content
    assert pathlib.Path(object_id).exists(), "Destination file should exist after successful async_put."
    with open(object_id, "rb") as f:
        received_content = f.read()
    assert received_content == expected_content, "Received file content does not match expected content."

    # Clean up
    safe_remove_file(object_id)


def test_successful_async_get(
    transfer_services: tuple[
        transfer_service_ext.TransferService,
        transfer_service_ext.TransferService,
        str,
        str,
    ],
) -> None:
    """Verifies a successful async_get operation between two services.

    Args:
        transfer_services: A pytest fixture providing two initialized services.
    """
    service1, service2, addr1, _ = transfer_services

    source_object_id = "source.object"
    dest_object_id = "dest.object"
    data_to_send = np.arange(20, dtype=np.int64)
    expected_content = data_to_send.tobytes()

    # Create the source file that service1 will "own"
    with open(source_object_id, "wb") as f:
        f.write(expected_content)

    # Call the async_get method to retrieve data from service1 to service2.
    get_future = service2.async_get(
        source_object_id,  # source_object_id
        addr1,  # source_address
        dest_object_id,  # dest_object_id
    )

    # Block until the C++ future completes and get the result.
    result = get_future.result(timeout=10)  # Add a 10 seconds timeout for safety.

    _LOGGER.info("GET Transfer result: %s", result)
    assert result.success is True, "The 'success' flag should be True on a successful GET transfer."
    assert not result.error_message, (
        f"The 'error_message' should be empty on a successful GET transfer, but was: '{result.error_message}'"
    )

    # Verify the content of the received file
    with open(dest_object_id, "rb") as f:
        received_content = f.read()
    assert received_content == expected_content, "Received file content does not match expected content."

    # Clean up
    safe_remove_file(source_object_id)
    safe_remove_file(dest_object_id)


def test_async_get_non_blocking(
    transfer_services: tuple[
        transfer_service_ext.TransferService,
        transfer_service_ext.TransferService,
        str,
        str,
    ],
) -> None:
    """Verifies that async_get is non-blocking and the transfer happens in the background.

    Args:
        transfer_services: A pytest fixture providing two initialized services.
    """
    service1, service2, addr1, _ = transfer_services

    source_object_id = "source_non_blocking_get.object"
    dest_object_id = "dest_non_blocking_get.object"
    data_to_send = np.arange(20, dtype=np.int64)
    expected_content = data_to_send.tobytes()

    # Create the source file that service1 will "own"
    with open(source_object_id, "wb") as f:
        f.write(expected_content)

    # Ensure the destination file does not exist before the test
    safe_remove_file(dest_object_id)

    # Initiate the async_get operation
    get_future = service2.async_get(
        source_object_id,  # source_object_id
        addr1,  # source_address
        dest_object_id,  # dest_object_id
    )

    # Immediately after calling async_get, assert that the destination file does NOT exist yet.
    # This proves the Python call itself is non-blocking.
    assert not pathlib.Path(dest_object_id).exists(), (
        "Destination file should not exist immediately after non-blocking async_get."
    )

    # Block until the C++ future completes and get the result.
    result = get_future.result(timeout=10)

    assert result.success is True, "The 'success' flag should be True on a successful GET transfer."
    assert not result.error_message, (
        f"The 'error_message' should be empty on a successful GET transfer, but was: '{result.error_message}'"
    )

    # Now, verify the file exists and its content
    assert pathlib.Path(dest_object_id).exists(), "Destination file should exist after successful async_get."
    with open(dest_object_id, "rb") as f:
        received_content = f.read()
    assert received_content == expected_content, "Received file content does not match expected content."

    # Clean up
    safe_remove_file(source_object_id)
    safe_remove_file(dest_object_id)


def test_async_put_invalid_size(
    transfer_services: tuple[
        transfer_service_ext.TransferService,
        transfer_service_ext.TransferService,
        str,
        str,
    ],
) -> None:
    """Verifies that async_put with an invalid size (0) fails.

    Args:
        transfer_services: A pytest fixture providing two initialized services.
    """
    # Given
    service1, _, _, addr2 = transfer_services

    object_id = "test_invalid_size.object"
    data_to_send = np.arange(20, dtype=np.int64)

    # When
    # Initiate the async_put operation with size 0
    put_future = service1.async_put(
        data_to_send.ctypes.data,  # data_ptr
        0,  # data_size is 0
        addr2,  # dest_address
        object_id,  # dest_object_id
    )

    # Then
    with pytest.raises(RuntimeError, match="Empty buffer or null data_ptr"):
        put_future.result(timeout=10)


def test_async_get_non_existent_object(
    transfer_services: tuple[
        transfer_service_ext.TransferService,
        transfer_service_ext.TransferService,
        str,
        str,
    ],
) -> None:
    """Verifies that async_get for a non-existent object fails.

    Args:
        transfer_services: A pytest fixture providing two initialized services.
    """
    # Given
    _, service2, addr1, _ = transfer_services

    source_object_id = "non_existent_source.object"
    dest_object_id = "dest.object"

    # When
    # Initiate the async_get operation
    get_future = service2.async_get(
        source_object_id,  # source_object_id
        addr1,  # source_address
        dest_object_id,  # dest_object_id
    )

    # Then
    with pytest.raises(RuntimeError, match="Received error message"):
        get_future.result(timeout=10)
