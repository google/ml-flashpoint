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

import pytest

from ml_flashpoint.core.checkpoint_id_types import (
    CheckpointContainerId,
    CheckpointObjectId,
)


class TestCheckpointObjectId:
    def test_valid_creation(self):
        obj_id = CheckpointObjectId("/valid/path/to/object")
        assert str(obj_id) == "/valid/path/to/object"

    def test_creation_with_trailing_slash(self):
        obj_id = CheckpointObjectId("/valid/path/to/object/")
        assert str(obj_id) == "/valid/path/to/object"

    def test_invalid_creation_no_leading_slash(self):
        with pytest.raises(ValueError):
            CheckpointObjectId("invalid/path")

    def test_invalid_creation_root(self):
        with pytest.raises(ValueError):
            CheckpointObjectId("/")

    def test_invalid_creation_root_repetitive(self):
        with pytest.raises(ValueError):
            CheckpointObjectId("//")
        with pytest.raises(ValueError):
            CheckpointObjectId("///")
        with pytest.raises(ValueError):
            CheckpointObjectId("////")
        with pytest.raises(ValueError):
            CheckpointObjectId("/////")

    def test_get_parent_default(self):
        obj_id = CheckpointObjectId("/a/b/c/d")
        parent = obj_id.get_parent()
        assert isinstance(parent, CheckpointContainerId)
        assert str(parent) == "/a/b/c"

    def test_get_parent_level_2(self):
        obj_id = CheckpointObjectId("/a/b/c/d")
        parent = obj_id.get_parent(levels_up=2)
        assert isinstance(parent, CheckpointContainerId)
        assert str(parent) == "/a/b"

    def test_get_parent_level_3(self):
        obj_id = CheckpointObjectId("/a/b/c/d")
        parent = obj_id.get_parent(levels_up=3)
        assert isinstance(parent, CheckpointContainerId)
        assert str(parent) == "/a"

    def test_get_parent_level_too_high_returns_none(self):
        obj_id = CheckpointObjectId("/a/b")
        parent = obj_id.get_parent(levels_up=2)
        assert parent is None

        obj_id = CheckpointObjectId("/a/b/c")
        parent = obj_id.get_parent(levels_up=3)
        assert parent is None

        obj_id = CheckpointObjectId("/a")
        parent = obj_id.get_parent(levels_up=1)
        assert parent is None

    def test_get_parent_invalid_level_zero(self):
        obj_id = CheckpointObjectId("/a/b/c")
        with pytest.raises(ValueError):
            obj_id.get_parent(levels_up=0)

    def test_get_parent_invalid_level_negative(self):
        obj_id = CheckpointObjectId("/a/b/c")
        with pytest.raises(ValueError):
            obj_id.get_parent(levels_up=-1)

    def test_string_representation(self):
        obj_id = CheckpointObjectId("/test/string")
        assert str(obj_id) == "/test/string"
        assert obj_id.data == "/test/string"

    def test_creation_with_multiple_slashes(self):
        # Empty segments are not allowed
        with pytest.raises(ValueError):
            CheckpointObjectId("/a//b")

    def test_invalid_creation_segment_ends_with_dot(self):
        with pytest.raises(ValueError, match="Path segment cannot end with a '.'"):
            CheckpointObjectId("/a/b./c")
        with pytest.raises(ValueError, match="Path segment cannot end with a '.'"):
            CheckpointObjectId("/a/b.")

    def test_equality(self):
        obj_id1 = CheckpointObjectId("/same/path")
        obj_id2 = CheckpointObjectId("/same/path")
        obj_id3 = CheckpointObjectId("/different/path")
        assert obj_id1 == obj_id2
        assert obj_id1 != obj_id3
        assert obj_id1 == "/same/path"
        assert obj_id1 != "/different/path"

    def test_from_container_valid(self):
        parent = CheckpointContainerId("/base/path")
        obj_id = CheckpointObjectId.from_container(parent, "object_name")
        assert str(obj_id) == "/base/path/object_name"
        assert isinstance(obj_id, CheckpointObjectId)

    def test_from_container_with_slashes(self):
        parent = CheckpointContainerId("/base/path/")
        obj_id = CheckpointObjectId.from_container(parent, "/object_name/")
        assert str(obj_id) == "/base/path/object_name"

    def test_from_container_invalid_parent_type(self):
        with pytest.raises(TypeError):
            CheckpointObjectId.from_container("/base/path", "object_name")

    def test_from_container_empty_object_name(self):
        parent = CheckpointContainerId("/base/path")
        with pytest.raises(ValueError):
            CheckpointObjectId.from_container(parent, "")

    def test_from_container_slash_object_name(self):
        parent = CheckpointContainerId("/base/path")
        with pytest.raises(ValueError):
            CheckpointObjectId.from_container(parent, "/")

    def test_from_container_special_chars_object_name(self):
        parent = CheckpointContainerId("/base/path")
        obj_id = CheckpointObjectId.from_container(parent, "obj-name.ext=1")
        assert str(obj_id) == "/base/path/obj-name.ext=1"

    def test_from_container_whitespace_object_name(self):
        parent = CheckpointContainerId("/base/path")
        with pytest.raises(ValueError):
            CheckpointObjectId.from_container(parent, "  ")

    def test_from_container_leading_trailing_whitespace_object_name(self):
        parent = CheckpointContainerId("/base/path")
        obj_id = CheckpointObjectId.from_container(parent, "  object_name  ")
        assert str(obj_id) == "/base/path/object_name"

    def test_from_container_dot_object_name(self):
        parent = CheckpointContainerId("/base/path")
        with pytest.raises(ValueError):
            CheckpointObjectId.from_container(parent, ".")

    def test_from_container_dotdot_object_name(self):
        parent = CheckpointContainerId("/base/path")
        with pytest.raises(ValueError):
            CheckpointObjectId.from_container(parent, "..")

    def test_from_container_multiple_dots_object_name(self):
        parent = CheckpointContainerId("/base/path")
        # Segments cannot end with a dot
        with pytest.raises(ValueError):
            CheckpointObjectId.from_container(parent, "...")

    def test_from_container_invalid_chars_object_name(self):
        parent = CheckpointContainerId("/base/path")
        # Invalid names are empty, single slash, multiple slashes, dot, dotdot
        with pytest.raises(ValueError):
            CheckpointObjectId.from_container(parent, "")
        with pytest.raises(ValueError):
            CheckpointObjectId.from_container(parent, "/")
        with pytest.raises(ValueError):
            CheckpointObjectId.from_container(parent, "//")
        with pytest.raises(ValueError):
            CheckpointObjectId.from_container(parent, ".")
        with pytest.raises(ValueError):
            CheckpointObjectId.from_container(parent, "..")


class TestCheckpointContainerId:
    def test_valid_creation(self):
        ver_id = CheckpointContainerId("/valid/path/to/version")
        assert str(ver_id) == "/valid/path/to/version"

    def test_creation_with_trailing_slash(self):
        ver_id = CheckpointContainerId("/valid/path/to/version/")
        assert str(ver_id) == "/valid/path/to/version"

    def test_invalid_creation_no_leading_slash(self):
        with pytest.raises(ValueError):
            CheckpointContainerId("invalid/path")

    @pytest.mark.parametrize("path", ["/", "//", "///"])
    def test_invalid_creation_root(self, path):
        with pytest.raises(ValueError, match="CheckpointContainerId cannot be the root path '/'"):
            CheckpointContainerId(path)

    def test_invalid_creation_segment_ends_with_dot(self):
        with pytest.raises(ValueError, match="Path segment cannot end with a '.'"):
            CheckpointContainerId("/a/b./c")
        with pytest.raises(ValueError, match="Path segment cannot end with a '.'"):
            CheckpointContainerId("/a/b.")

    def test_invalid_creation_empty_segment(self):
        with pytest.raises(ValueError, match="Path segment cannot be empty"):
            CheckpointContainerId("/a//b")

    def test_get_parent_default(self):
        ver_id = CheckpointContainerId("/a/b/c")
        parent = ver_id.get_parent()
        assert isinstance(parent, CheckpointContainerId)
        assert str(parent) == "/a/b"

    def test_get_parent_root_is_none(self):
        ver_id = CheckpointContainerId("/a")
        parent = ver_id.get_parent()
        assert parent is None

        ver_id = CheckpointContainerId("/a/b")
        parent = ver_id.get_parent(levels_up=2)
        assert parent is None

    def test_equality(self):
        ver_id1 = CheckpointContainerId("/same/path")
        ver_id2 = CheckpointContainerId("/same/path")
        ver_id3 = CheckpointContainerId("/different/path")
        assert ver_id1 == ver_id2
        assert ver_id1 != ver_id3
        assert ver_id1 == "/same/path"
        assert ver_id1 != "/different/path"

    def test_create_child_valid(self):
        parent = CheckpointContainerId("/base/path")
        child = CheckpointContainerId.create_child(parent, "child")
        assert str(child) == "/base/path/child"

    def test_create_child_with_slashes(self):
        parent = CheckpointContainerId("/base/path/")
        child = CheckpointContainerId.create_child(parent, "/child/")
        assert str(child) == "/base/path/child"

    def test_create_child_empty_child(self):
        parent = CheckpointContainerId("/base/path")
        child = CheckpointContainerId.create_child(parent, "")
        assert str(child) == "/base/path"

    def test_create_child_slash_child(self):
        parent = CheckpointContainerId("/base/path")
        child = CheckpointContainerId.create_child(parent, "/")
        assert str(child) == "/base/path"

    def test_create_child_invalid_parent_type(self):
        with pytest.raises(TypeError):
            CheckpointContainerId.create_child("/base/path", "child")

    def test_get_version_container_pattern(self):
        # When
        pattern = CheckpointContainerId.get_version_container_pattern()
        # Then
        assert pattern.match("step-123_ckpt")
        assert not pattern.match("step-123_checkpoint")
        assert not pattern.match("mystep-123_ckpt")

    def test_get_dirty_version_checkpoint_container_pattern(self):
        # When
        pattern = CheckpointContainerId.get_dirty_version_checkpoint_container_pattern()
        # Then
        assert pattern.match("step-123_ckpt_unfinished")
        assert pattern.match("step-123_ckpt_rank0__unfinished")
        assert pattern.match("step-123_ckpt_rank0_unfinished")
        assert not pattern.match("step-123_ckpt")
        assert not pattern.match("step-123_checkpoint")
        assert not pattern.match("mystep-123_ckpt")

    def test_format_version_container(self):
        # When
        formatted_string = CheckpointContainerId.format_version_container(123)
        # Then
        assert formatted_string == "step-123_ckpt"

    def test_parse_version_container_step(self):
        # When
        step = CheckpointContainerId.parse_version_container_step("step-123_ckpt")
        # Then
        assert step == 123

    def test_parse_version_container_step_invalid(self):
        # When
        step = CheckpointContainerId.parse_version_container_step("invalid-step")
        # Then
        assert step is None

    def test_format_version_container_negative_step(self):
        # When
        formatted_string = CheckpointContainerId.format_version_container(-1)
        # Then
        assert formatted_string == "step--1_ckpt"

    def test_format_version_container_float_step(self):
        # When
        formatted_string = CheckpointContainerId.format_version_container(123.0)
        # Then
        assert formatted_string == "step-123.0_ckpt"

    def test_parse_version_container_step_almost_matching(self):
        # When
        step = CheckpointContainerId.parse_version_container_step("step-123_ckpt_extra")
        # Then
        assert step is None

    def test_parse_version_container_step_negative_string(self):
        # When
        step = CheckpointContainerId.parse_version_container_step("step--1_ckpt")
        # Then
        assert step is None

    def test_parse_version_container_step_non_string_input(self):
        # When/Then
        with pytest.raises(TypeError):
            CheckpointContainerId.parse_version_container_step(123)
        with pytest.raises(TypeError):
            CheckpointContainerId.parse_version_container_step(None)
