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

import json
import os

import pytest
from assertpy import assert_that

from ml_flashpoint.adapter.megatron_bridge import local_checkpoint_index as index_module
from ml_flashpoint.adapter.megatron_bridge.local_checkpoint_index import (
    NO_CHECKPOINT,
    MLFlashpointLocalCheckpointIndex,
)
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId


@pytest.fixture
def base_container(tmp_path) -> CheckpointContainerId:
    return CheckpointContainerId(str(tmp_path / "mlf"))


@pytest.fixture(autouse=True)
def _single_rank(mocker):
    """Runs the index as if this were the only, node-local-zero rank."""
    mocker.patch.object(index_module.dist, "is_initialized", return_value=True)
    mocker.patch.object(index_module.dist, "get_node_local_rank", return_value=0)


def _make_container(base_container: CheckpointContainerId, step: int) -> CheckpointContainerId:
    container = CheckpointContainerId.create_child(base_container, CheckpointContainerId.format_version_container(step))
    os.makedirs(str(container), exist_ok=True)
    return container


class TestFindLatest:
    def test_returns_sentinel_when_nothing_recoverable(self, base_container, mocker):
        # Given
        loader = mocker.MagicMock()
        loader.get_latest_complete_checkpoint.return_value = None
        index = MLFlashpointLocalCheckpointIndex(base_container, loader)

        # When
        result = index.find_latest()

        # Then
        assert_that(result).is_equal_to(NO_CHECKPOINT)

    def test_returns_step_of_latest_container(self, base_container, mocker):
        # Given
        container = _make_container(base_container, 120)
        loader = mocker.MagicMock()
        loader.get_latest_complete_checkpoint.return_value = container
        index = MLFlashpointLocalCheckpointIndex(base_container, loader)

        # When
        result = index.find_latest()

        # Then
        assert_that(result).is_equal_to(120)

    def test_returns_sentinel_for_unparseable_container_name(self, base_container, mocker):
        # Given
        container = CheckpointContainerId(str(base_container) + "/not-a-version-dir")
        os.makedirs(str(container), exist_ok=True)
        loader = mocker.MagicMock()
        loader.get_latest_complete_checkpoint.return_value = container
        index = MLFlashpointLocalCheckpointIndex(base_container, loader)

        # When
        result = index.find_latest()

        # Then
        assert_that(result).is_equal_to(NO_CHECKPOINT)

    def test_discovery_runs_once(self, base_container, mocker):
        # Given
        loader = mocker.MagicMock()
        loader.get_latest_complete_checkpoint.return_value = _make_container(base_container, 5)
        index = MLFlashpointLocalCheckpointIndex(base_container, loader)

        # When
        index.find_latest()
        index.find_latest()
        index.resolve_latest_container()

        # Then
        assert_that(loader.get_latest_complete_checkpoint.call_count).is_equal_to(1)

    def test_discovery_failure_is_swallowed(self, base_container, mocker):
        # Given
        loader = mocker.MagicMock()
        loader.get_latest_complete_checkpoint.side_effect = RuntimeError("peer unreachable")
        index = MLFlashpointLocalCheckpointIndex(base_container, loader)

        # When
        result = index.find_latest()

        # Then
        assert_that(result).is_equal_to(NO_CHECKPOINT)

    def test_invalidate_forces_rediscovery(self, base_container, mocker):
        # Given
        loader = mocker.MagicMock()
        loader.get_latest_complete_checkpoint.return_value = None
        index = MLFlashpointLocalCheckpointIndex(base_container, loader)
        index.find_latest()

        # When
        index.invalidate()
        index.find_latest()

        # Then
        assert_that(loader.get_latest_complete_checkpoint.call_count).is_equal_to(2)

    def test_disable_reports_no_checkpoint(self, base_container, mocker):
        # Given
        loader = mocker.MagicMock()
        loader.get_latest_complete_checkpoint.return_value = _make_container(base_container, 7)
        index = MLFlashpointLocalCheckpointIndex(base_container, loader)

        # When
        index.disable()

        # Then
        assert_that(index.find_latest()).is_equal_to(NO_CHECKPOINT)
        assert_that(index.resolve_latest_container()).is_none()
        loader.get_latest_complete_checkpoint.assert_not_called()


class TestMetadataStub:
    def test_writes_stub_for_discovered_container(self, base_container, mocker):
        # Given
        container = _make_container(base_container, 42)
        loader = mocker.MagicMock()
        loader.get_latest_complete_checkpoint.return_value = container
        index = MLFlashpointLocalCheckpointIndex(base_container, loader)

        # When
        index.resolve_latest_container()

        # Then
        stub_path = os.path.join(str(container), "metadata.json")
        assert_that(os.path.exists(stub_path)).is_true()
        with open(stub_path) as handle:
            assert_that(json.load(handle)).is_equal_to({"sharded_backend": ""})

    def test_does_not_overwrite_existing_stub(self, base_container, mocker):
        # Given
        container = _make_container(base_container, 42)
        stub_path = os.path.join(str(container), "metadata.json")
        with open(stub_path, "w") as handle:
            json.dump({"sharded_backend": "already-here"}, handle)
        loader = mocker.MagicMock()
        loader.get_latest_complete_checkpoint.return_value = container
        index = MLFlashpointLocalCheckpointIndex(base_container, loader)

        # When
        index.resolve_latest_container()

        # Then
        with open(stub_path) as handle:
            assert_that(json.load(handle)).is_equal_to({"sharded_backend": "already-here"})

    def test_non_local_rank_zero_does_not_write(self, base_container, mocker):
        # Given
        mocker.patch.object(index_module.dist, "get_node_local_rank", return_value=1)
        container = _make_container(base_container, 42)
        loader = mocker.MagicMock()
        loader.get_latest_complete_checkpoint.return_value = container
        index = MLFlashpointLocalCheckpointIndex(base_container, loader)

        # When
        index.resolve_latest_container()

        # Then
        assert_that(os.path.exists(os.path.join(str(container), "metadata.json"))).is_false()

    def test_write_failure_does_not_propagate(self, base_container, mocker):
        # Given a container path that was never created on disk.
        container = CheckpointContainerId(str(base_container) + "/step-1_ckpt")
        loader = mocker.MagicMock()
        loader.get_latest_complete_checkpoint.return_value = container
        index = MLFlashpointLocalCheckpointIndex(base_container, loader)

        # When
        resolved = index.resolve_latest_container()

        # Then
        assert_that(resolved).is_equal_to(container)


class TestProperties:
    def test_local_ckpt_dir_is_base_container(self, base_container, mocker):
        # Given
        index = MLFlashpointLocalCheckpointIndex(base_container, mocker.MagicMock())

        # When/Then
        assert_that(index.local_ckpt_dir).is_equal_to(str(base_container))

    def test_latest_container_is_none_before_discovery(self, base_container, mocker):
        # Given
        index = MLFlashpointLocalCheckpointIndex(base_container, mocker.MagicMock())

        # When/Then
        assert_that(index.latest_container).is_none()

    def test_latest_container_after_discovery(self, base_container, mocker):
        # Given
        container = _make_container(base_container, 3)
        loader = mocker.MagicMock()
        loader.get_latest_complete_checkpoint.return_value = container
        index = MLFlashpointLocalCheckpointIndex(base_container, loader)

        # When
        index.resolve_latest_container()

        # Then
        assert_that(index.latest_container).is_equal_to(container)
