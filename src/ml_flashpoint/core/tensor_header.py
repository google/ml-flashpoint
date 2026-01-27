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

import dataclasses
import json
import struct
from typing import Tuple


@dataclasses.dataclass
class TensorHeader:
    """Header for a tensor stored in the checkpoint buffer."""

    dtype: str
    shape: list[int]

    def to_bytes(self) -> bytes:
        """Serializes the header to bytes.

        Format:
        [4 bytes HEADER_LEN] [HEADER_BYTES (JSON)]
        """
        metadata = {
            "dtype": self.dtype,
            "shape": self.shape,
        }
        json_bytes = json.dumps(metadata).encode("utf-8")
        header_len = len(json_bytes)

        return struct.pack("<I", header_len) + json_bytes

    @classmethod
    def from_bytes(cls, buffer: bytes) -> Tuple["TensorHeader", int]:
        """Deserializes the header from bytes.

        Args:
            buffer: The bytes to deserialize.

        Returns:
            A tuple containing the TensorHeader and the number of bytes consumed.
        """
        # Read header length
        header_len = struct.unpack("<I", buffer[:4])[0]

        # Read header
        json_bytes = buffer[4 : 4 + header_len]
        metadata = json.loads(json_bytes.decode("utf-8"))

        return cls(dtype=metadata["dtype"], shape=metadata["shape"]), 4 + header_len
