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
import uuid


def generate_hfid(prefix: str) -> str:
    """Returns a human-friendly ID (HFID) in the form of "<prefix>_<unique ID>".

    The prefix must not be empty, and must contain only lowercase alphanumeric
    characters. The unique ID is alphanumeric.

    Args:
        prefix: The prefix for the ID.

    Returns:
        The generated human-friendly ID.

    Raises:
        ValueError: If the prefix is empty or contains invalid characters.
    """
    if not prefix:
        raise ValueError("Prefix cannot be empty.")
    if not re.match(r"^[a-z0-9]+$", prefix):
        raise ValueError("Prefix must contain only lowercase alphanumeric characters.")

    return f"{prefix}_{uuid.uuid4().hex}"
