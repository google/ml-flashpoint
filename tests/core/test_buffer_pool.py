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
from concurrent.futures import ThreadPoolExecutor
from typing import Generator

import pytest

from ml_flashpoint.checkpoint_object_manager.buffer_io import BufferIO
from ml_flashpoint.checkpoint_object_manager.buffer_metadata import METADATA_SIZE
from ml_flashpoint.core.buffer_pool import BufferIOProxy, BufferPool


class TestBufferPool:
    @pytest.fixture
    def buffer_pool_config(self, tmp_path) -> dict:
        pool_dir = tmp_path / ".buffer_pool"
        return {
            "pool_dir_path": str(pool_dir),
            "rank": 0,
            "num_buffers": 3,
            "buffer_size": METADATA_SIZE + 1024,
        }

    @pytest.fixture
    def buffer_pool(self, buffer_pool_config) -> Generator[BufferPool, None, None]:
        # Reset singleton
        BufferPool._instance = None

        pool = BufferPool(**buffer_pool_config)
        yield pool

        pool.teardown()
        BufferPool._instance = None

    def test_acquire_preallocated_buffer(self, buffer_pool, buffer_pool_config):
        """Verifies that acquire reuses pre-allocated buffers."""
        buffer_io = buffer_pool.acquire()
        assert buffer_io is not None
        assert isinstance(buffer_io, BufferIOProxy)

        # Check that the buffer path follows the naming convention
        buffer_id = buffer_io.buffer_obj.get_id()
        assert "buffer_0_" in buffer_id
        assert os.path.exists(buffer_id)

        # Verify it was removed from free_buffers
        assert len(buffer_pool.free_buffers) == buffer_pool_config["num_buffers"] - 1
        assert len(buffer_pool.active_buffers) == 1

        # Cleanup
        buffer_io.close()
        assert buffer_io.closed
        assert not buffer_io._buffer_io.closed

    def test_reuse_lazy_resize_buffer(self, buffer_pool, tmp_path, buffer_pool_config):
        """Verifies that acquire does NOT resize eagerly, but write auto-resizes."""
        symlink_path = str(tmp_path / "symlink_1")
        buffer_size = buffer_pool_config["buffer_size"]

        # 1. Acquire
        buf_io1 = buffer_pool.acquire(associated_symlink=symlink_path)

        # Write data using buffer object instead of truncating file
        buf_io1.write(b"dummy")

        current_cap = buf_io1.buffer_obj.get_capacity()
        assert current_cap == buffer_size

        # 2. Release via GC
        os.remove(symlink_path)
        buffer_pool._gc()

        # 3. Acquire again
        buf_io2 = buffer_pool.acquire()

        assert buf_io1._buffer_io is buf_io2._buffer_io
        # Verify capacity did NOT increase yet (lazy)
        assert buf_io2.buffer_obj.get_capacity() == buffer_size

        # 4. Write data that fits -> NO resize
        buf_io2.write(b"x" * 100)
        assert buf_io2.buffer_obj.get_capacity() == buffer_size

        # 5. Write data that exceeds -> Auto Resize
        large_data = b"y" * (buffer_size + 100)
        buf_io2.write(large_data)
        assert buf_io2.buffer_obj.get_capacity() > buffer_size

    def test_gc_releases_orphaned_buffers(self, buffer_pool, tmp_path):
        """Verifies that GC correctly releases buffers with deleted symlinks."""
        symlink_path = str(tmp_path / "my_symlink")

        # Acquire with symlink=None
        buf_io = buffer_pool.acquire(associated_symlink=None)

        # Verify IT IS ACTIVE
        assert buf_io._buffer_io in buffer_pool.active_buffers

        # Trigger GC - should NOT release (pending registration)
        buffer_pool._gc()
        assert buf_io._buffer_io in buffer_pool.active_buffers

        # Re-acquire with symlink
        buf_io.close()

        # Create valid symlink path
        buf_io = buffer_pool.acquire(associated_symlink=symlink_path)

        assert os.path.islink(symlink_path)
        assert buf_io._buffer_io in buffer_pool.active_buffers

        # Delete symlink
        os.remove(symlink_path)

        # Trigger GC
        buffer_pool._gc()

        assert buf_io._buffer_io not in buffer_pool.active_buffers
        assert buf_io._buffer_io in buffer_pool.free_buffers

    def test_pool_exhaustion(self, buffer_pool, buffer_pool_config, tmp_path):
        """Verifies that acquiring more than num_buffers raises RuntimeError."""
        buffers = []
        # Acquire all buffers
        for _ in range(buffer_pool_config["num_buffers"]):
            buffers.append(buffer_pool.acquire())

        # Try one more
        with pytest.raises(RuntimeError, match="BufferPool exhausted"):
            buffer_pool.acquire()

        # Cleanup
        for b in buffers:
            b.close()

    def test_proxy_truncates_on_close(self, buffer_pool):
        """Verifies that closing the proxy truncates the underlying buffer."""
        buf_proxy = buffer_pool.acquire()

        data = b"hello world"
        buf_proxy.write(data)

        initial_capacity = buf_proxy.buffer_obj.get_capacity()
        assert initial_capacity > 0

        buf_proxy.close()

        assert buf_proxy.closed
        real_buf = buf_proxy._buffer_io
        assert not real_buf.closed

        expected_size = METADATA_SIZE + len(data)
        assert real_buf.buffer_obj.get_capacity() == expected_size

    def test_proxy_auto_resizes(self, buffer_pool, buffer_pool_config):
        """Verifies that the proxy auto-resizes when writing beyond capacity."""
        buf_proxy = buffer_pool.acquire()
        buffer_size = buffer_pool_config["buffer_size"]

        # 1. Write data that fits
        buf_proxy.write(b"a" * 50)
        assert buf_proxy.buffer_obj.get_capacity() == buffer_size

        # 2. Write data that EXCEEDS capacity
        large_data = b"b" * (buffer_size + 1000)
        buf_proxy.write(large_data)

        current_capacity = buf_proxy.buffer_obj.get_capacity()
        assert current_capacity > buffer_size

        # 3. Test next_buffer_slice resize
        huge_size = current_capacity * 2
        mv = buf_proxy.next_buffer_slice(huge_size)
        assert len(mv) == huge_size

        final_capacity = buf_proxy.buffer_obj.get_capacity()
        assert final_capacity > current_capacity
        buf_proxy.close()

    def test_reuse_with_resize_resets_content(self, buffer_pool, tmp_path, buffer_pool_config):
        """Verifies that reusing a buffer with resize resets position and content."""
        symlink_path = str(tmp_path / "symlink_resize_test")

        # 1. Acquire and write data
        buf_io1 = buffer_pool.acquire(associated_symlink=symlink_path)

        data1 = b"FirstData"
        buf_io1.write(data1)
        assert buf_io1.tell() == len(data1)

        # 2. Release via GC
        os.remove(symlink_path)
        buffer_pool._gc()

        # 3. Acquire again
        buf_io2 = buffer_pool.acquire()

        assert buf_io1._buffer_io is buf_io2._buffer_io
        assert buf_io2.tell() == 0

        # Write new data
        data2 = b"SecondData"
        buf_io2.write(data2)

        # Verify content
        buf_io2.seek(0)
        content = buf_io2.read()
        assert content == data2

    def test_init_with_invalid_dir(self, mocker):
        """Verifies behavior when initialization fails due to directory issues."""
        BufferPool._instance = None
        mocker.patch("os.makedirs", side_effect=OSError("Permission denied"))
        with pytest.raises(OSError, match="Permission denied"):
            BufferPool(pool_dir_path="/invalid/path", num_buffers=1, buffer_size=1024)
        BufferPool._instance = None

    def test_init_zero_size(self, tmp_path):
        """Verifies behavior when initialized with size 0."""
        BufferPool._instance = None
        # buffer_size=0 should skip preallocation
        pool = BufferPool(pool_dir_path=str(tmp_path), num_buffers=3, buffer_size=0)
        assert len(pool.free_buffers) == 0
        pool.teardown()
        BufferPool._instance = None

    def test_concurrent_acquire(self, buffer_pool):
        """Verifies that acquire is thread-safe."""

        # We have 3 buffers. Try to acquire 3 concurrently.
        def acquire_task():
            try:
                return buffer_pool.acquire()
            except RuntimeError:
                return None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(acquire_task) for _ in range(5)]
            results = [f.result() for f in futures]

        # Should have 3 successes and 2 failures (None)
        successes = [r for r in results if r is not None]
        assert len(successes) == 3

        # Cleanup
        for b in successes:
            b.close()

    def test_symlink_creation_failure(self, buffer_pool, tmp_path):
        """Verifies that acquire fails and releases buffer if symlink creation fails."""
        symlink_path = str(tmp_path / "symlink_fail")
        # Create a directory at symlink path to cause OSError
        os.makedirs(symlink_path)

        with pytest.raises(OSError):
            buffer_pool.acquire(associated_symlink=symlink_path)

        # Verify buffer was returned to free pool
        assert len(buffer_pool.free_buffers) == 3
        assert len(buffer_pool.active_buffers) == 0

    def test_init_missing_pool_dir(self):
        """Verifies that initialization fails if pool_dir_path is missing."""
        BufferPool._instance = None
        with pytest.raises(TypeError, match="missing 1 required positional argument: 'pool_dir_path'"):
            BufferPool(rank=0, num_buffers=3, buffer_size=1024)
        BufferPool._instance = None


class TestBufferIOProxy:
    @pytest.fixture
    def mock_buffer_io(self, mocker):
        """Creates a mock BufferIO object."""
        buffer_io = mocker.Mock(spec=BufferIO)
        buffer_io.closed = False
        # Setup buffer_obj mock for capacity checks
        buffer_io.buffer_obj = mocker.Mock()
        buffer_io.buffer_obj.get_capacity.return_value = 1000
        buffer_io.tell.return_value = 0
        return buffer_io

    @pytest.fixture
    def proxy(self, mock_buffer_io):
        """Creates a BufferIOProxy wrapping the mock."""
        return BufferIOProxy(mock_buffer_io)

    def test_delegation_basic(self, proxy, mock_buffer_io):
        """Verifies that basic methods are delegated to the underlying BufferIO."""
        # read
        proxy.read(10)
        mock_buffer_io.read.assert_called_with(10)

        # seek
        proxy.seek(5, 1)
        mock_buffer_io.seek.assert_called_with(5, 1)

        # tell
        proxy.tell()
        mock_buffer_io.tell.assert_called_once()

        # flush
        proxy.flush()
        mock_buffer_io.flush.assert_called_once()

    def test_properties(self, proxy, mock_buffer_io):
        """Verifies property delegation."""
        # buffer_obj
        assert proxy.buffer_obj is mock_buffer_io.buffer_obj

        # closed
        assert not proxy.closed
        mock_buffer_io.closed = True
        assert proxy.closed

    def test_write_delegation_success(self, proxy, mock_buffer_io):
        """Verifies write delegates correctly when no resize is needed."""
        data = b"test"
        proxy.write(data)
        mock_buffer_io.write.assert_called_with(data)

    def test_write_auto_resize(self, proxy, mock_buffer_io):
        """Verifies write triggers auto-resize when BufferIO raises ValueError."""
        # Setup mock to raise ValueError on first write, then succeed
        mock_buffer_io.write.side_effect = [
            ValueError("exceeds buffer capacity"),
            10,  # Return bytes written on retry
        ]

        # Current capacity 1000, current pos 0
        mock_buffer_io.buffer_obj.get_capacity.return_value = 1000
        mock_buffer_io.tell.return_value = 0

        data = b"x" * 1500  # Needs more than 1000

        # Call write
        proxy.write(data)

        # Verify resize was called
        # Calculation: max(1000 * 1.1, METADATA_SIZE + 0 + 1500 + PADDING_SIZE)
        assert mock_buffer_io.resize.called
        # Check that it called write again
        assert mock_buffer_io.write.call_count == 2
        mock_buffer_io.write.assert_called_with(data)

    def test_next_buffer_slice_delegation_success(self, proxy, mock_buffer_io):
        """Verifies next_buffer_slice delegates correctly."""
        proxy.next_buffer_slice(100)
        mock_buffer_io.next_buffer_slice.assert_called_with(100)

    def test_next_buffer_slice_auto_resize(self, proxy, mock_buffer_io, mocker):
        """Verifies next_buffer_slice triggers resize."""
        mock_buffer_io.next_buffer_slice.side_effect = [
            ValueError("exceeds buffer capacity"),
            mocker.Mock(),  # Return a mock slice on retry
        ]

        mock_buffer_io.buffer_obj.get_capacity.return_value = 1000
        mock_buffer_io.tell.return_value = 900

        # Request 200 bytes (total 1100 > 1000)
        proxy.next_buffer_slice(200)

        assert mock_buffer_io.resize.called
        assert mock_buffer_io.next_buffer_slice.call_count == 2

    def test_close_truncate(self, proxy, mock_buffer_io, mocker):
        """Verifies close calls buffer_obj.resize if truncate is True."""
        mock_buffer_io.is_readonly = False
        mock_buffer_io._metadata = mocker.Mock()
        mock_buffer_io._metadata.len_written_data = 500
        mock_buffer_io._mv = range(1000)  # Mock len()

        proxy.close(truncate=True)

        target = METADATA_SIZE + 500
        mock_buffer_io.resize.assert_called_with(target)

        assert proxy.closed

    def test_close_no_truncate(self, proxy, mock_buffer_io):
        """Verifies close does not resize if truncate is False."""
        proxy.close(truncate=False)
        mock_buffer_io.resize.assert_not_called()
        assert proxy.closed
