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
import time
import multiprocessing as mp
import pytest

from ml_flashpoint.checkpoint_object_manager.buffer_object.buffer_object_ext import BufferPool

def worker_target(shm_name, pool_dir, rank, num_buffers, buffer_size, result_queue):
    """Worker process that tries to acquire a buffer."""
    try:
        # Attach to pool
        pool = BufferPool(shm_name=shm_name, pool_dir=pool_dir, rank=rank,
                          num_buffers=num_buffers, buffer_size=buffer_size)
        
        try:
             buf_path = pool.acquire()
             result_queue.put(("SUCCESS", buf_path))
        except RuntimeError as e:
             result_queue.put(("ERROR", str(e)))
             
    except Exception as e:
        result_queue.put(("EXCEPTION", str(e)))

class TestCppBufferPoolMultiprocess:
    def test_shared_pool_exhaustion(self, tmp_path):
        """Verifies that pool is shared across processes and exhaustion is respected."""
        # Use a unique shm name to avoid conflicts with running tests
        shm_name = f"/test_shm_pool_{int(time.time())}"
        pool_dir = str(tmp_path / "pool_dir")
        os.makedirs(pool_dir)
        rank = 0
        num_buffers = 2
        buffer_size = 1024
        
        # 1. Initialize pool in main process
        pool = BufferPool(shm_name=shm_name, pool_dir=pool_dir, rank=rank,
                          num_buffers=num_buffers, buffer_size=buffer_size)
        
        # 2. Acquire all buffers in main process
        path1 = pool.acquire()
        path2 = pool.acquire()
        
        # 3. Start worker process to try to acquire another buffer
        result_queue = mp.Queue()
        p = mp.Process(target=worker_target, args=(shm_name, pool_dir, rank, num_buffers, buffer_size, result_queue))
        p.start()
        
        p.join(timeout=5)
        if p.is_alive():
             p.terminate()
             pytest.fail("Worker process timed out")
             
        status, message = result_queue.get()
        assert status == "ERROR"
        assert "BufferPool exhausted" in message
        
        # 4. Release one buffer in main process
        pool.release(path1)
        
        # 5. Start another worker process
        p2 = mp.Process(target=worker_target, args=(shm_name, pool_dir, rank, num_buffers, buffer_size, result_queue))
        p2.start()
        p2.join(timeout=5)
        
        status, message = result_queue.get()
        assert status == "SUCCESS"
        assert message == path1 # Should get the released buffer

    def test_buffer_resize_via_symlink(self, tmp_path):
        """Verifies that BufferObject resizes the pooled buffer when opened via symlink with overwrite=True."""
        shm_name = f"/test_shm_pool_resize_{int(time.time())}"
        pool_dir = str(tmp_path / "pool_dir")
        os.makedirs(pool_dir)
        rank = 0
        num_buffers = 1
        buffer_size = 1024
        
        pool = BufferPool(shm_name=shm_name, pool_dir=pool_dir, rank=rank,
                          num_buffers=num_buffers, buffer_size=buffer_size)
        
        symlink_path = str(tmp_path / "my_symlink")
        path1 = pool.acquire(symlink_path)
        
        assert os.path.islink(symlink_path)
        
        target_path = os.readlink(symlink_path)
        assert os.path.getsize(target_path) == buffer_size
        
        new_size = 2048
        from ml_flashpoint.checkpoint_object_manager.buffer_object.buffer_object_ext import BufferObject
        
        buffer_obj = BufferObject(symlink_path, new_size, True)
        
        assert os.path.getsize(target_path) == new_size
        
        buffer_obj.close()
        pool.release(symlink_path)
