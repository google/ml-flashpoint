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

import re
from enum import Enum

DIRTY_MARKER_SUFFIX = "unfinished"
GLOBAL_RANK_PATTERN = re.compile(r"src(\d+)")
COMMON_STATE_FNAME = "common.pt"


class CheckpointFormat(bytes, Enum):
    # Standard PyTorch save format
    TORCH_SAVE = b"TORCH___"
    # Our custom optimized format
    MLF_FORMAT = b"MLF_TENS"


FORMAT_SIGNATURE = CheckpointFormat.MLF_FORMAT.value


def default_metadata_object_name() -> str:
    """Returns the default object name for metadata files (i.e. filename).

    Returns:
        The default object name as a string.
    """
    return ".metadata"
