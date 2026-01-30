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

# --- Metadata Definitions ---
METADATA_SIZE = 4096  # 4KB


class BufferMetadataType(ctypes.LittleEndianStructure):
    """Defines the structure of the metadata block stored at the beginning
    of the BufferIO buffer.
    """

    _pack_ = 1  # Ensure tight packing for cross-platform consistency
    _fields_ = [
        # 8 bytes for the length of valid data written *after* the metadata block
        ("len_written_data", ctypes.c_uint64),
        # 8 bytes for checkpoint format signature to identify the file format version
        ("format_signature", ctypes.c_char * 8),
        # Pad the rest of the structure to reach METADATA_SIZE
        (
            "reserved",
            ctypes.c_uint8 * (METADATA_SIZE - ctypes.sizeof(ctypes.c_uint64) - 8),
        ),
    ]


# --- Sanity Check ---
# Ensure the total size of the structure matches the defined METADATA_SIZE
assert ctypes.sizeof(BufferMetadataType) == METADATA_SIZE, (
    f"BufferMetadataType size mismatch: Actual {ctypes.sizeof(BufferMetadataType)} != Expected {METADATA_SIZE}"
)


# --- Helper Function ---
def get_metadata_str(metadata: BufferMetadataType | None) -> str:
    """Returns a string representation of the BufferMetadataType object.

    Args:
        metadata: An instance of BufferMetadataType or None.

    Returns:
        A formatted string describing the metadata content, or a placeholder
        if metadata is None.
    """
    if metadata is None:
        return "[Metadata: None]"
    # Format the fields from the metadata object
    # Add more fields here if the BufferMetadataType struct expands
    len_data = metadata.len_written_data
    # You could add other relevant fields from 'reserved' if they were defined
    return f"Metadata(len_written_data={len_data})"
