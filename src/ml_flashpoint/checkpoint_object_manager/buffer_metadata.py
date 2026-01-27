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

import ctypes
import dataclasses
import pickle

import torch

METADATA_SIZE = 4096


@dataclasses.dataclass
class TensorHeader:
    """Header information for a stored tensor."""

    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


class BufferMetadataType(ctypes.LittleEndianStructure):
    """Metadata block stored at the beginning of the BufferIO buffer."""

    _pack_ = 1
    _fields_ = [
        ("len_written_data", ctypes.c_uint64),
        ("reserved", ctypes.c_uint8 * (METADATA_SIZE - ctypes.sizeof(ctypes.c_uint64))),
    ]

    @property
    def tensor_manifest(self) -> dict[int, TensorHeader]:
        try:
            reserved_bytes = bytes(self.reserved)
            return pickle.loads(reserved_bytes) if any(reserved_bytes) else {}
        except Exception:
            return {}

    @tensor_manifest.setter
    def tensor_manifest(self, manifest: dict[int, TensorHeader]):
        data = pickle.dumps(manifest)
        if len(data) > ctypes.sizeof(self.reserved):
            raise ValueError(f"Manifest too large ({len(data)} > {ctypes.sizeof(self.reserved)})")
        ctypes.memset(ctypes.addressof(self.reserved), 0, ctypes.sizeof(self.reserved))
        ctypes.memmove(ctypes.addressof(self.reserved), data, len(data))


assert ctypes.sizeof(BufferMetadataType) == METADATA_SIZE, "Metadata size mismatch"


def get_metadata_str(metadata: BufferMetadataType | None) -> str:
    return f"Metadata(len_written_data={metadata.len_written_data})" if metadata else "[Metadata: None]"
