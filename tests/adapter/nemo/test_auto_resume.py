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
from typing import Any

import pytest
from nemo.lightning import AutoResume
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from pytest_mock import MockerFixture

from ml_flashpoint.adapter.nemo.auto_resume import MLFlashpointAutoResume
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.checkpoint_loader import (
    DefaultMLFlashpointCheckpointLoader,
    MLFlashpointCheckpointLoader,
)
from ml_flashpoint.replication.replication_manager import ReplicationManager


@pytest.fixture
def mock_checkpoint_loader(mocker: MockerFixture) -> Any:
    """Provides a mock MLFlashpointCheckpointLoader instance."""
    return mocker.create_autospec(MLFlashpointCheckpointLoader, instance=True)


@pytest.fixture
def checkpoint_base_container() -> CheckpointContainerId:
    """Provides a consistent CheckpointContainerId for the base path."""
    return CheckpointContainerId("/tmp/ml_flashpoint_checkpoints")


@pytest.fixture
def auto_resume(
    mock_checkpoint_loader: Any,
    checkpoint_base_container: CheckpointContainerId,
) -> MLFlashpointAutoResume:
    """Provides an MLFlashpointAutoResume instance with mocked dependencies."""
    return MLFlashpointAutoResume(
        checkpoint_loader=mock_checkpoint_loader,
        checkpoint_base_container=checkpoint_base_container,
    )


class TestMLFlashpointAutoResume:
    """Tests the functionality of the MLFlashpointAutoResume class."""

    def test_initializer(self, mocker):
        """Ensures dependency imports and class instantiation are successful."""
        chkpt_obj_manager = CheckpointObjectManager()
        replication_manager = ReplicationManager()
        checkpoint_loader = DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=chkpt_obj_manager,
            replication_manager=replication_manager,
        )
        base_container = CheckpointContainerId("/tmp/ml_flashpoint_checkpoints")

        # Act
        auto_resume = MLFlashpointAutoResume(
            checkpoint_loader=checkpoint_loader,
            checkpoint_base_container=base_container,
        )

        # Assert
        assert auto_resume.checkpoint_loader == checkpoint_loader
        assert auto_resume.checkpoint_base_container == base_container

    def test_initializer_superclass_properties_are_correct(self, mocker):
        """Tests that init sets the parent class's properties correctly."""
        # Arrange
        checkpoint_loader = DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=CheckpointObjectManager(),
            replication_manager=ReplicationManager(),
        )
        base_container = CheckpointContainerId("/tmp/ml_flashpoint_checkpoints")

        # Act
        auto_resume = MLFlashpointAutoResume(
            checkpoint_loader=checkpoint_loader,
            checkpoint_base_container=base_container,
        )

        # Assert
        # NeMo's AutoResume defaults are False
        assert auto_resume.resume_if_exists is False
        assert auto_resume.resume_ignore_no_checkpoint is False

    def test_initializer_propagates_true_params(self):
        """Tests that init respects the passed parameters when they are True."""
        # Arrange
        checkpoint_loader = DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=CheckpointObjectManager(),
            replication_manager=ReplicationManager(),
        )
        base_container = CheckpointContainerId("/tmp/ml_flashpoint_checkpoints")

        # Act
        auto_resume = MLFlashpointAutoResume(
            checkpoint_loader=checkpoint_loader,
            checkpoint_base_container=base_container,
            resume_if_exists=True,
            resume_ignore_no_checkpoint=True,
        )

        # Assert
        assert auto_resume.resume_if_exists is True
        assert auto_resume.resume_ignore_no_checkpoint is True

    def test_initializer_respects_params(self):
        """Tests that init respects the passed parameters for resume flags."""
        # Arrange
        checkpoint_loader = DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=CheckpointObjectManager(), replication_manager=ReplicationManager()
        )
        base_container = CheckpointContainerId("/tmp/ml_flashpoint_checkpoints")

        # Act
        auto_resume = MLFlashpointAutoResume(
            checkpoint_loader=checkpoint_loader,
            checkpoint_base_container=base_container,
            resume_if_exists=False,
            resume_ignore_no_checkpoint=False,
        )

        # Assert
        assert auto_resume.resume_if_exists is False
        assert auto_resume.resume_ignore_no_checkpoint is False

    def test_initializer_passes_kwargs_to_super(self):
        """Tests that kwargs (like restore_config) are passed to the superclass."""
        # Arrange
        checkpoint_loader = DefaultMLFlashpointCheckpointLoader(
            checkpoint_object_manager=CheckpointObjectManager(), replication_manager=ReplicationManager()
        )
        base_container = CheckpointContainerId("/tmp/ml_flashpoint_checkpoints")
        restore_config = RestoreConfig(path="nemo://some-model")

        # Act
        auto_resume = MLFlashpointAutoResume(
            checkpoint_loader=checkpoint_loader,
            checkpoint_base_container=base_container,
            restore_config=restore_config,
        )

        # Assert
        assert auto_resume.restore_config == restore_config

    def test_get_trainer_ckpt_path_finds_mlf_checkpoint(
        self,
        auto_resume: MLFlashpointAutoResume,
        mock_checkpoint_loader: Any,
        checkpoint_base_container: CheckpointContainerId,
        mocker: MockerFixture,
    ):
        """
        Tests that it returns the ML Flashpoint checkpoint path when one is found.
        """
        # Arrange
        auto_resume.resume_if_exists = True
        expected_id = CheckpointContainerId.create_child(checkpoint_base_container, "step-100_ckpt")
        mock_checkpoint_loader.get_latest_complete_checkpoint.return_value = expected_id
        mock_super_find = mocker.patch.object(AutoResume, "_find_trainer_ckpt_path")
        mocker.patch("ml_flashpoint.adapter.nemo.auto_resume.dist.get_node_local_rank", return_value=0)
        mock_exists = mocker.patch("ml_flashpoint.adapter.nemo.auto_resume.os.path.exists", return_value=True)

        # Act
        result = auto_resume.get_trainer_ckpt_path()

        # Assert
        assert result == Path(expected_id.data)
        mock_checkpoint_loader.get_latest_complete_checkpoint.assert_called_once_with(checkpoint_base_container)
        # Verify that the parent class's method was NOT called.
        mock_super_find.assert_not_called()
        mock_exists.assert_called_once()

    def test_get_trainer_ckpt_path_uses_cache(
        self,
        auto_resume: MLFlashpointAutoResume,
        mock_checkpoint_loader: Any,
        checkpoint_base_container: CheckpointContainerId,
        mocker: MockerFixture,
    ):
        """
        Tests that it uses the cached path on subsequent calls.
        """
        # Arrange
        auto_resume.resume_if_exists = True
        expected_id = CheckpointContainerId.create_child(checkpoint_base_container, "step-100_ckpt")
        mock_checkpoint_loader.get_latest_complete_checkpoint.return_value = expected_id
        mocker.patch("ml_flashpoint.adapter.nemo.auto_resume.dist.get_node_local_rank", return_value=0)
        mocker.patch("ml_flashpoint.adapter.nemo.auto_resume.os.path.exists", return_value=True)

        # Act
        result1 = auto_resume.get_trainer_ckpt_path()
        result2 = auto_resume.get_trainer_ckpt_path()

        # Assert
        assert result1 == Path(expected_id.data)
        assert result2 == Path(expected_id.data)
        # Should only be called ONCE despite two calls
        mock_checkpoint_loader.get_latest_complete_checkpoint.assert_called_once_with(checkpoint_base_container)

    def test_get_trainer_ckpt_path_creates_metadata_if_missing(
        self,
        auto_resume: MLFlashpointAutoResume,
        mock_checkpoint_loader: Any,
        mocker: MockerFixture,
        tmp_path: Path,
    ):
        """
        Tests that metadata.json is created if it doesn't exist and local_rank is 0.
        """
        # Arrange
        # Use a real temporary directory for the checkpoint
        base_path = tmp_path / "checkpoints"
        base_path.mkdir()
        checkpoint_base_container = CheckpointContainerId(str(base_path))

        # Update auto_resume to use the temp path
        auto_resume.checkpoint_base_container = checkpoint_base_container

        expected_id = CheckpointContainerId.create_child(checkpoint_base_container, "step-100_ckpt")
        # Create the directory for the checkpoint so we can write metadata.json to it
        Path(expected_id.data).mkdir()

        mock_checkpoint_loader.get_latest_complete_checkpoint.return_value = expected_id
        mocker.patch("ml_flashpoint.adapter.nemo.auto_resume.dist.get_node_local_rank", return_value=0)
        auto_resume.resume_if_exists = True

        # Act
        auto_resume.get_trainer_ckpt_path()

        # Assert
        metadata_path = Path(expected_id.data) / "metadata.json"
        assert metadata_path.exists()
        assert metadata_path.read_text() == '{"sharded_backend": ""}'

    def test_get_trainer_ckpt_path_skips_metadata_if_exists(
        self,
        auto_resume: MLFlashpointAutoResume,
        mock_checkpoint_loader: Any,
        mocker: MockerFixture,
        tmp_path: Path,
    ):
        """
        Tests that metadata.json creation is skipped if it already exists.
        """
        # Arrange
        # Use a real temporary directory for the checkpoint
        base_path = tmp_path / "checkpoints"
        base_path.mkdir()
        checkpoint_base_container = CheckpointContainerId(str(base_path))

        # Update auto_resume to use the temp path
        auto_resume.checkpoint_base_container = checkpoint_base_container

        expected_id = CheckpointContainerId.create_child(checkpoint_base_container, "step-100_ckpt")
        # Create the directory for the checkpoint
        checkpoint_dir = Path(expected_id.data)
        checkpoint_dir.mkdir()

        # Create a pre-existing metadata.json with specific content
        metadata_path = checkpoint_dir / "metadata.json"
        original_content = '{"sharded_backend": "existing"}'
        metadata_path.write_text(original_content)

        mock_checkpoint_loader.get_latest_complete_checkpoint.return_value = expected_id
        mocker.patch("ml_flashpoint.adapter.nemo.auto_resume.dist.get_node_local_rank", return_value=0)
        auto_resume.resume_if_exists = True

        # Act
        auto_resume.get_trainer_ckpt_path()

        # Assert
        # Verify the content hasn't changed
        assert metadata_path.read_text() == original_content

    def test_get_trainer_ckpt_path_skips_metadata_if_not_rank0(
        self,
        auto_resume: MLFlashpointAutoResume,
        mock_checkpoint_loader: Any,
        checkpoint_base_container: CheckpointContainerId,
        mocker: MockerFixture,
    ):
        """
        Tests that metadata.json logic is skipped if local_rank is not 0.
        """
        # Arrange
        expected_id = CheckpointContainerId.create_child(checkpoint_base_container, "step-100_ckpt")
        mock_checkpoint_loader.get_latest_complete_checkpoint.return_value = expected_id
        mocker.patch("ml_flashpoint.adapter.nemo.auto_resume.dist.get_node_local_rank", return_value=1)
        auto_resume.resume_if_exists = True
        mock_exists = mocker.patch("ml_flashpoint.adapter.nemo.auto_resume.os.path.exists")

        # Act
        auto_resume.get_trainer_ckpt_path()

        # Assert
        mock_exists.assert_not_called()

    def test_get_trainer_ckpt_path_falls_back_to_super(
        self,
        auto_resume: MLFlashpointAutoResume,
        mock_checkpoint_loader: Any,
        checkpoint_base_container: CheckpointContainerId,
        mocker: MockerFixture,
    ):
        """
        Tests that it falls back to the parent method if no MLF checkpoint is found.
        """
        # Arrange
        auto_resume.resume_if_exists = True
        mock_checkpoint_loader.get_latest_complete_checkpoint.return_value = None
        fallback_path = Path("/tmp/nemo_checkpoints/fallback.ckpt")
        mock_super_find = mocker.patch.object(AutoResume, "_find_trainer_ckpt_path", return_value=fallback_path)

        # Act
        result = auto_resume.get_trainer_ckpt_path()

        # Assert
        assert result == fallback_path
        mock_checkpoint_loader.get_latest_complete_checkpoint.assert_called_once_with(checkpoint_base_container)
        # Verify that the parent class's method WAS called as a fallback.
        mock_super_find.assert_called_once()

    def test_get_trainer_ckpt_path_only_calls_resolve_once_even_if_none(
        self,
        auto_resume: MLFlashpointAutoResume,
        mock_checkpoint_loader: Any,
        checkpoint_base_container: CheckpointContainerId,
        mocker: MockerFixture,
    ):
        """
        Tests that resolution is attempted only once even if it returns None.
        """
        # Arrange
        auto_resume.resume_if_exists = True
        mock_checkpoint_loader.get_latest_complete_checkpoint.return_value = None
        mocker.patch.object(AutoResume, "_find_trainer_ckpt_path", return_value=None)

        # Act
        auto_resume.get_trainer_ckpt_path()
        auto_resume.get_trainer_ckpt_path()

        # Assert
        # Should be called exactly once
        mock_checkpoint_loader.get_latest_complete_checkpoint.assert_called_once_with(checkpoint_base_container)
