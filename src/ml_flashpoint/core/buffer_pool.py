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


from dataclasses import dataclass


@dataclass
class BufferPoolConfig:
    """Configuration for the BufferPool."""

    pool_dir_path: str
    rank: int
    num_buffers: int
    buffer_size: int

    def __post_init__(self):
        if not self.pool_dir_path:
            raise ValueError("pool_dir_path must be provided in BufferPoolConfig")
        if not isinstance(self.rank, int) or self.rank < 0:
            raise ValueError("rank must be a non-negative integer in BufferPoolConfig")
        if not isinstance(self.num_buffers, int) or self.num_buffers < 0:
            raise ValueError("num_buffers must be a non-negative integer in BufferPoolConfig")
        if not isinstance(self.buffer_size, int) or self.buffer_size < 0:
            raise ValueError("buffer_size must be a non-negative integer in BufferPoolConfig")
