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

from pathlib import Path

import pytest

from ml_flashpoint.checkpoint_object_manager.object_manager import object_manager_ext


def test_delete_directories_async(tmp_path: Path):
    # Given
    dir1 = tmp_path / "dir1"
    dir2 = tmp_path / "dir2"
    dir1.mkdir()
    dir2.mkdir()
    dirs_to_delete = [str(dir1), str(dir2)]

    # When
    future = object_manager_ext.delete_directories_async(dirs_to_delete)
    future.wait()  # Block until deletion is complete.

    # Then
    assert not dir1.exists()
    assert not dir2.exists()


def test_delete_directories_async_empty_list():
    # Given
    dirs_to_delete = []

    # When/Then: This should not raise an exception.
    try:
        future = object_manager_ext.delete_directories_async(dirs_to_delete)
        future.wait()
    except Exception as e:
        pytest.fail(f"delete_directories_async raised an exception with an empty list: {e}")


def test_delete_directories_async_with_files(tmp_path: Path):
    # Given
    file1_path = tmp_path / "file1.txt"
    file1_path.write_text("test")
    paths_to_delete = [str(file1_path)]

    # When
    future = object_manager_ext.delete_directories_async(paths_to_delete)
    future.wait()

    # Then
    assert file1_path.exists()
