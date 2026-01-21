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

"""Tests for HFID: human-friendly ID generation."""

import re

import pytest

from ml_flashpoint.core.hfid import generate_hfid


class TestGenerateHfid:
    def test_valid_prefix(self):
        prefix = "test"
        hfid = generate_hfid(prefix)
        assert hfid.startswith(f"{prefix}_")
        uuid_part = hfid.split("_")[1]
        assert re.match(r"^[0-9a-f]{32}$", uuid_part) is not None

    def test_format(self):
        prefix = "example"
        hfid = generate_hfid(prefix)
        parts = hfid.split("_")
        assert len(parts) == 2
        assert parts[0] == prefix
        assert len(parts[1]) == 32  # UUID hex length

    def test_uuid_is_hex(self):
        prefix = "check"
        hfid = generate_hfid(prefix)
        uuid_part = hfid.split("_")[1]
        assert re.match(r"^[0-9a-f]{32}$", uuid_part) is not None

    def test_uniqueness(self):
        prefix = "unique"
        hfid1 = generate_hfid(prefix)
        hfid2 = generate_hfid(prefix)
        assert hfid1 != hfid2

    def test_empty_prefix_raises_error(self):
        with pytest.raises(ValueError, match="Prefix cannot be empty."):
            generate_hfid("")

    def test_prefix_with_uppercase_raises_error(self):
        with pytest.raises(ValueError, match="Prefix must contain only lowercase alphanumeric characters."):
            generate_hfid("Test")

    def test_prefix_with_hyphen_raises_error(self):
        with pytest.raises(ValueError, match="Prefix must contain only lowercase alphanumeric characters."):
            generate_hfid("test-prefix")

    def test_prefix_with_slash_raises_error(self):
        with pytest.raises(ValueError, match="Prefix must contain only lowercase alphanumeric characters."):
            generate_hfid("test/prefix")

    def test_prefix_with_numbers_is_valid(self):
        prefix = "test123"
        hfid = generate_hfid(prefix)
        assert hfid.startswith(f"{prefix}_")
        uuid_part = hfid.split("_")[1]
        assert re.match(r"^[0-9a-f]{32}$", uuid_part) is not None
