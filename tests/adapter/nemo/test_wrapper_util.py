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

"""Tests for the NeMo wrapper utility."""

import dataclasses

import pytest
from nemo import lightning as nl
from nemo.lightning.io.pl import MegatronCheckpointIO
from nemo.lightning.pytorch import strategies as nl_strategies
from nemo.lightning.pytorch import trainer as nl_trainer
from nemo.utils.callbacks.dist_ckpt_io import AsyncFinalizableCheckpointIO

from ml_flashpoint.adapter.nemo.auto_resume import MLFlashpointAutoResume
from ml_flashpoint.adapter.nemo.checkpoint_callback import MLFlashpointCheckpointCallback
from ml_flashpoint.adapter.nemo.checkpoint_io import (
    MLFlashpointAsyncFinalizableCheckpointIO,
    MLFlashpointCheckpointIO,
)
from ml_flashpoint.adapter.nemo.wrapper_util import (
    wrap_trainer_and_auto_resume_with_mlflashpoint,
    wrap_trainer_checkpoint_io_with_mlflashpoint,
)
from ml_flashpoint.adapter.pytorch.memory_storage_writer import MemoryStorageWriter
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.checkpoint_loader import DefaultMLFlashpointCheckpointLoader
from ml_flashpoint.core.checkpoint_saver import (
    DEFAULT_INITIAL_BUFFER_SIZE_BYTES,
)
from ml_flashpoint.replication.replication_manager import ReplicationManager


class TestWrapTrainerAndAutoResumeWithMLFlashpoint:
    """Tests for the wrap_trainer_and_auto_resume_with_mlflashpoint function."""

    @pytest.fixture
    def mock_ckpt_obj_manager(self, mocker):
        return mocker.MagicMock(spec=CheckpointObjectManager)

    @pytest.fixture
    def mock_replication_manager(self, mocker):
        return mocker.MagicMock(spec=ReplicationManager)

    def test_successful_wrap_and_resume_creation(self, mocker):
        """Tests the successful creation of MLFlashpointAutoResume and wrapping."""
        # Given
        # Mock the heavy components
        mock_replication_manager_cls = mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.ReplicationManager")
        mock_replication_manager_instance = mock_replication_manager_cls.return_value

        mock_wrap_trainer = mocker.patch(
            "ml_flashpoint.adapter.nemo.wrapper_util.wrap_trainer_checkpoint_io_with_mlflashpoint"
        )

        # Inputs
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        flashpoint_base_container = "/tmp/test_container"
        async_save = True

        # Use a mock for default_auto_resume.
        # We assume for this test that MLFlashpointAutoResume can be instantiated
        # with just the arguments we provide + empty vars from the default.
        default_auto_resume = nl.AutoResume(
            resume_from_path="/some/test/resume/path", resume_if_exists=True, resume_ignore_no_checkpoint=True
        )

        # When
        actual_auto_resume = wrap_trainer_and_auto_resume_with_mlflashpoint(
            trainer, flashpoint_base_container, async_save, default_auto_resume
        )

        # Then
        # 1. ReplicationManager initialized
        mock_replication_manager_cls.assert_called_once()
        mock_replication_manager_instance.initialize.assert_called_once()
        # Capture the ckpt_obj_manager passed to initialize
        _, kwargs_init = mock_replication_manager_instance.initialize.call_args
        ckpt_obj_manager = kwargs_init["checkpoint_object_manager"]
        assert isinstance(ckpt_obj_manager, CheckpointObjectManager)

        # 2. wrap_trainer_checkpoint_io_with_mlflashpoint called
        mock_wrap_trainer.assert_called_once_with(
            trainer=trainer,
            flashpoint_base_container=flashpoint_base_container,
            ckpt_obj_manager=ckpt_obj_manager,  # Same instance
            replication_manager=mock_replication_manager_instance,
            async_save=async_save,
            checkpoint_loader=actual_auto_resume.checkpoint_loader,
            always_save_context=False,
            write_thread_count=1,
            initial_write_buffer_size_bytes=DEFAULT_INITIAL_BUFFER_SIZE_BYTES,
            use_optimized_save=True,
            use_cached_ckpt_structure=False,
        )

        # 3. Result is correct type and has correct attributes
        assert isinstance(actual_auto_resume, MLFlashpointAutoResume)
        assert actual_auto_resume.checkpoint_base_container == flashpoint_base_container
        assert isinstance(actual_auto_resume.checkpoint_loader, DefaultMLFlashpointCheckpointLoader)
        for field in dataclasses.fields(default_auto_resume):
            assert getattr(default_auto_resume, field.name) == getattr(actual_auto_resume, field.name)
        # Verify the loader has the same object manager
        assert actual_auto_resume.checkpoint_loader._checkpoint_object_manager is ckpt_obj_manager

    def test_successful_wrap_with_none_default_auto_resume(self, mocker):
        """Tests successful wrapping when default_auto_resume is None."""
        # Given
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.ReplicationManager")
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.wrap_trainer_checkpoint_io_with_mlflashpoint")
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        flashpoint_base_container = "/tmp/test_container"

        # When
        actual_auto_resume = wrap_trainer_and_auto_resume_with_mlflashpoint(
            trainer, flashpoint_base_container, async_save=True, default_auto_resume=None
        )

        # Then
        assert isinstance(actual_auto_resume, MLFlashpointAutoResume)
        assert actual_auto_resume.checkpoint_base_container == CheckpointContainerId(flashpoint_base_container)
        # Verify that other attributes are set to defaults (since we passed None)
        # We can check a default attribute of AutoResume, e.g., resume_if_exists is False by default
        assert actual_auto_resume.resume_if_exists is False

    @pytest.mark.parametrize("flashpoint_base_container", ["", None])
    def test_validation_missing_base_container(self, mocker, flashpoint_base_container):
        """Tests validation check for missing base container."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        default_auto_resume = mocker.MagicMock(spec=nl.AutoResume)

        # When/Then
        with pytest.raises(ValueError, match="The 'flashpoint_base_container' argument cannot be empty"):
            wrap_trainer_and_auto_resume_with_mlflashpoint(
                trainer, flashpoint_base_container, async_save=True, default_auto_resume=default_auto_resume
            )

    @pytest.mark.parametrize(
        "flashpoint_base_container_input",
        ["/tmp/test_container", CheckpointContainerId("/tmp/test_container")],
    )
    def test_container_id_types(self, mocker, flashpoint_base_container_input):
        """Tests that both str and CheckpointContainerId are accepted."""
        # Given
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.ReplicationManager")
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.wrap_trainer_checkpoint_io_with_mlflashpoint")

        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        default_auto_resume = nl.AutoResume()

        # When
        actual_auto_resume = wrap_trainer_and_auto_resume_with_mlflashpoint(
            trainer, flashpoint_base_container_input, async_save=True, default_auto_resume=default_auto_resume
        )

        # Then
        # Verify that the container ID is correctly set in the result
        expected_container_id = CheckpointContainerId(str(flashpoint_base_container_input))
        assert actual_auto_resume.checkpoint_base_container == expected_container_id

    @pytest.mark.parametrize(
        "buffer_size_kwarg, expected_buffer_size",
        [
            ({}, DEFAULT_INITIAL_BUFFER_SIZE_BYTES),
            ({"initial_write_buffer_size_bytes": 12345}, 12345),
        ],
    )
    def test_initial_save_buffer_size_forwarding(
        self, mocker, mock_ckpt_obj_manager, mock_replication_manager, buffer_size_kwarg, expected_buffer_size
    ):
        """Tests that the initial_save_buffer_size_bytes is forwarded correctly."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        original_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        trainer.strategy.checkpoint_io = original_checkpoint_io

        # Patch the saver to check the arguments passed to it
        mock_saver = mocker.patch(
            "ml_flashpoint.adapter.nemo.wrapper_util.DefaultMLFlashpointCheckpointSaver",
        )

        # Mocks and inputs
        flashpoint_base_container = "/test_base_container"
        async_save = True
        default_auto_resume = nl.AutoResume()
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.ReplicationManager")

        # When
        wrap_trainer_and_auto_resume_with_mlflashpoint(
            trainer,
            flashpoint_base_container,
            async_save,
            default_auto_resume,
            **buffer_size_kwarg,
        )

        # Then
        assert isinstance(trainer.strategy.checkpoint_io, MLFlashpointCheckpointIO)
        # Verify that the saver was initialized with the correct buffer size
        mock_saver.assert_called_once()
        _, kwargs = mock_saver.call_args
        assert kwargs["initial_buffer_size_bytes"] == expected_buffer_size

    @pytest.mark.parametrize(
        "thread_count_kwarg, expected_thread_count",
        [
            ({}, 1),
            ({"write_thread_count": 4}, 4),
        ],
    )
    def test_write_thread_count_forwarding(
        self, mocker, mock_ckpt_obj_manager, mock_replication_manager, thread_count_kwarg, expected_thread_count
    ):
        """Tests that the write_thread_count is forwarded correctly."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        original_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        trainer.strategy.checkpoint_io = original_checkpoint_io

        # Spy on the MemoryStorageWriter's __init__ method
        spy_memory_storage_writer_init = mocker.spy(MemoryStorageWriter, "__init__")

        # Mocks and inputs
        flashpoint_base_container = "/test_base_container"
        async_save = True
        default_auto_resume = nl.AutoResume()
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.ReplicationManager")

        # When
        wrap_trainer_and_auto_resume_with_mlflashpoint(
            trainer,
            flashpoint_base_container,
            async_save,
            default_auto_resume,
            **thread_count_kwarg,
        )

        # Then
        # Verify that MemoryStorageWriter was initialized with the correct thread count
        spy_memory_storage_writer_init.assert_called_once()
        _, kwargs = spy_memory_storage_writer_init.call_args  # Capture kwargs
        assert kwargs["thread_count"] == expected_thread_count

    @pytest.mark.parametrize("always_save_context", [True, False])
    def test_loader_initialization_arguments(self, mocker, always_save_context):
        """Tests that NeMoMLFlashpointCheckpointLoader is initialized with correct arguments."""
        # Given
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.ReplicationManager")
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.wrap_trainer_checkpoint_io_with_mlflashpoint")
        mock_nemo_checkpoint_loader_cls = mocker.patch(
            "ml_flashpoint.adapter.nemo.wrapper_util.NeMoMLFlashpointCheckpointLoader"
        )

        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        flashpoint_base_container = "/tmp/test_container"
        default_auto_resume = nl.AutoResume()

        # When
        wrap_trainer_and_auto_resume_with_mlflashpoint(
            trainer,
            flashpoint_base_container,
            async_save=True,
            default_auto_resume=default_auto_resume,
            always_save_context=always_save_context,
        )

        # Then
        mock_nemo_checkpoint_loader_cls.assert_called_once()
        _, kwargs = mock_nemo_checkpoint_loader_cls.call_args
        assert kwargs["recover_context"] == always_save_context


class TestWrapTrainerCheckpointIOWithMLFlashpoint:
    """Tests for the wrap_trainer_checkpoint_io_with_mlflashpoint function."""

    @pytest.fixture
    def mock_ckpt_obj_manager(self, mocker):
        return mocker.MagicMock(spec=CheckpointObjectManager)

    @pytest.fixture
    def mock_replication_manager(self, mocker):
        return mocker.MagicMock(spec=ReplicationManager)

    class TestParameterValidationChecks:
        """Tests for parameter validation."""

        def test_validation_missing_trainer(self, mocker, mock_ckpt_obj_manager, mock_replication_manager):
            """Tests validation check for missing trainer."""
            base_container = "/test_base_container"
            with pytest.raises(ValueError, match="The 'trainer' argument cannot be None"):
                wrap_trainer_checkpoint_io_with_mlflashpoint(
                    None,
                    base_container,
                    mock_ckpt_obj_manager,
                    mock_replication_manager,
                    async_save=True,
                    checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
                )

        def test_validation_missing_base_container(self, mocker, mock_ckpt_obj_manager, mock_replication_manager):
            """Tests validation check for missing base container."""
            trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
            with pytest.raises(ValueError, match="The 'flashpoint_base_container' argument cannot be empty"):
                wrap_trainer_checkpoint_io_with_mlflashpoint(
                    trainer,
                    "",
                    mock_ckpt_obj_manager,
                    async_save=True,
                    replication_manager=mock_replication_manager,
                    checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
                )

        def test_validation_missing_ckpt_obj_manager(self, mocker, mock_replication_manager):
            """Tests validation check for missing checkpoint object manager."""
            trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
            base_container = "/test_base_container"
            with pytest.raises(ValueError, match="The 'ckpt_obj_manager' argument cannot be None"):
                wrap_trainer_checkpoint_io_with_mlflashpoint(
                    trainer,
                    base_container,
                    None,
                    replication_manager=mock_replication_manager,
                    async_save=True,
                    checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
                )

        def test_validation_missing_replication_manager(self, mocker, mock_ckpt_obj_manager):
            """Tests validation check for missing replication manager."""
            trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
            base_container = "/test_base_container"
            with pytest.raises(ValueError, match="The 'replication_manager' argument cannot be None"):
                wrap_trainer_checkpoint_io_with_mlflashpoint(
                    trainer,
                    base_container,
                    mock_ckpt_obj_manager,
                    replication_manager=None,
                    async_save=True,
                    checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
                )

        def test_validation_invalid_write_thread_count(self, mocker, mock_ckpt_obj_manager, mock_replication_manager):
            """Tests validation check for invalid write thread count."""
            trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
            base_container = "/test_base_container"
            with pytest.raises(ValueError, match="write_thread_count must be >= 1"):
                wrap_trainer_checkpoint_io_with_mlflashpoint(
                    trainer,
                    base_container,
                    mock_ckpt_obj_manager,
                    replication_manager=mock_replication_manager,
                    async_save=True,
                    write_thread_count=0,
                    checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
                )

        def test_validation_invalid_initial_write_buffer_size(
            self, mocker, mock_ckpt_obj_manager, mock_replication_manager
        ):
            """Tests validation check for invalid initial write buffer size."""
            trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
            base_container = "/test_base_container"
            with pytest.raises(ValueError, match="initial_write_buffer_size_bytes must be > 0"):
                wrap_trainer_checkpoint_io_with_mlflashpoint(
                    trainer,
                    base_container,
                    mock_ckpt_obj_manager,
                    replication_manager=mock_replication_manager,
                    async_save=True,
                    initial_write_buffer_size_bytes=0,
                    checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
                )

        def test_validation_missing_checkpoint_loader(self, mocker, mock_ckpt_obj_manager, mock_replication_manager):
            """Tests validation check for missing checkpoint loader."""
            trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
            base_container = "/test_base_container"
            with pytest.raises(ValueError, match="The 'checkpoint_loader' argument cannot be None"):
                wrap_trainer_checkpoint_io_with_mlflashpoint(
                    trainer,
                    base_container,
                    mock_ckpt_obj_manager,
                    replication_manager=mock_replication_manager,
                    async_save=True,
                    checkpoint_loader=None,
                )

    def test_mlflashpoint_not_enabled(self, mocker, mock_ckpt_obj_manager, mock_replication_manager):
        """Tests that the function returns early if no MLF callback is found."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = []
        original_checkpoint_io = mocker.MagicMock()
        trainer.strategy.checkpoint_io = original_checkpoint_io
        base_container = "/test_base_container"

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            replication_manager=mock_replication_manager,
            async_save=True,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
        )

        # Then
        assert trainer.strategy.checkpoint_io is original_checkpoint_io

    def test_unsupported_strategy(self, mocker, mock_ckpt_obj_manager, mock_replication_manager):
        """Tests that a ValueError is raised for non-Megatron strategies."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.FSDPStrategy)
        base_container = "/test_base_container"

        # When/Then
        with pytest.raises(ValueError, match="Only MegatronStrategy is supported"):
            wrap_trainer_checkpoint_io_with_mlflashpoint(
                trainer,
                base_container,
                mock_ckpt_obj_manager,
                mock_replication_manager,
                async_save=True,
                checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
            )

    def test_unsupported_checkpoint_io_type(self, mocker, mock_ckpt_obj_manager, mock_replication_manager):
        """Tests that a ValueError is raised for non-MegatronCheckpointIO."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        trainer.strategy.checkpoint_io = mocker.MagicMock()  # Not MegatronCheckpointIO
        base_container = "/test_base_container"

        # When/Then
        with pytest.raises(
            ValueError,
            match="Expected checkpoint_io to be of type 'MegatronCheckpointIO'",
        ):
            wrap_trainer_checkpoint_io_with_mlflashpoint(
                trainer,
                base_container,
                mock_ckpt_obj_manager,
                mock_replication_manager,
                async_save=True,
                checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
            )

    def test_successful_wrapping_no_async_wrapper(self, mocker, mock_ckpt_obj_manager, mock_replication_manager):
        """Tests successful wrapping when no async wrapper is present."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        original_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        trainer.strategy.checkpoint_io = original_checkpoint_io
        base_container = "/test_base_container"

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            replication_manager=mock_replication_manager,
            async_save=True,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
        )

        # Then
        assert isinstance(trainer.strategy.checkpoint_io, MLFlashpointCheckpointIO)
        assert trainer.strategy.checkpoint_io.fallback_checkpoint_io is original_checkpoint_io
        assert trainer.strategy.checkpoint_io.async_save is True

    def test_successful_wrapping_with_async_wrapper(self, mocker, mock_ckpt_obj_manager, mock_replication_manager):
        """Tests successful wrapping when an async wrapper is present."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        inner_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        original_checkpoint_io = AsyncFinalizableCheckpointIO(inner_checkpoint_io)
        trainer.strategy.checkpoint_io = original_checkpoint_io
        base_container = "/test_base_container"

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            replication_manager=mock_replication_manager,
            async_save=True,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
        )

        # Then
        assert isinstance(trainer.strategy.checkpoint_io, MLFlashpointAsyncFinalizableCheckpointIO)
        wrapped_io = trainer.strategy.checkpoint_io.checkpoint_io
        assert isinstance(wrapped_io, MLFlashpointCheckpointIO)
        assert wrapped_io.fallback_checkpoint_io is inner_checkpoint_io
        assert wrapped_io.async_save is True

    def test_idempotency_check_no_async_wrapper(self, mocker, mock_ckpt_obj_manager, mock_replication_manager):
        """Tests that wrapping does not occur twice when async_save is False."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        original_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        trainer.strategy.checkpoint_io = original_checkpoint_io
        base_container = "/test_base_container"

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            replication_manager=mock_replication_manager,
            async_save=False,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
        )
        first_wrap_result = trainer.strategy.checkpoint_io
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            replication_manager=mock_replication_manager,
            async_save=False,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
        )
        second_wrap_result = trainer.strategy.checkpoint_io

        # Then
        assert first_wrap_result is second_wrap_result
        assert isinstance(second_wrap_result, MLFlashpointCheckpointIO)
        assert second_wrap_result.fallback_checkpoint_io is original_checkpoint_io

    def test_idempotency_check_with_mlf_async_wrapper_and_async_save_true(
        self, mocker, mock_ckpt_obj_manager, mock_replication_manager
    ):
        """Tests idempotency when the IO is already async-wrapped with MLF and async_save is True."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        base_container = "/test_base_container"

        # Create an already-wrapped MLFlashpointCheckpointIO
        inner_megatron_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        mlf_io = mocker.MagicMock(spec=MLFlashpointCheckpointIO)
        mlf_io.fallback_checkpoint_io = inner_megatron_io
        original_async_wrapped_mlf_io = MLFlashpointAsyncFinalizableCheckpointIO(mlf_io)

        trainer.strategy.checkpoint_io = original_async_wrapped_mlf_io

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            replication_manager=mock_replication_manager,
            async_save=True,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
        )
        first_wrap_result = trainer.strategy.checkpoint_io
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            replication_manager=mock_replication_manager,
            async_save=True,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
        )
        second_wrap_result = trainer.strategy.checkpoint_io

        # Then
        # The function should see the outer MLFlashpointAsyncFinalizableCheckpointIO and return,
        # leaving the original async-wrapped object untouched.
        assert first_wrap_result is second_wrap_result
        assert first_wrap_result is original_async_wrapped_mlf_io
        assert isinstance(first_wrap_result, MLFlashpointAsyncFinalizableCheckpointIO)
        assert second_wrap_result.checkpoint_io is mlf_io
        assert second_wrap_result.checkpoint_io.fallback_checkpoint_io is inner_megatron_io

    def test_successful_wrapping_with_async_save_false_no_async_wrapper(
        self, mocker, mock_ckpt_obj_manager, mock_replication_manager
    ):
        """Tests wrapping with async_save=False and no async wrapper."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        original_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        trainer.strategy.checkpoint_io = original_checkpoint_io
        base_container = "/test_base_container"

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            replication_manager=mock_replication_manager,
            async_save=False,  # Test False
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
        )

        # Then
        assert isinstance(trainer.strategy.checkpoint_io, MLFlashpointCheckpointIO)
        assert trainer.strategy.checkpoint_io.fallback_checkpoint_io is original_checkpoint_io
        assert trainer.strategy.checkpoint_io.async_save is False

    def test_successful_wrapping_with_async_save_false_with_async_wrapper(
        self, mocker, mock_ckpt_obj_manager, mock_replication_manager
    ):
        """Tests wrapping with async_save=False and an async wrapper."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        inner_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        original_checkpoint_io = AsyncFinalizableCheckpointIO(inner_checkpoint_io)
        trainer.strategy.checkpoint_io = original_checkpoint_io
        base_container = "/test_base_container"

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            replication_manager=mock_replication_manager,
            async_save=False,  # Test False
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
        )

        # Then
        assert isinstance(trainer.strategy.checkpoint_io, MLFlashpointAsyncFinalizableCheckpointIO)
        wrapped_io = trainer.strategy.checkpoint_io.checkpoint_io
        assert isinstance(wrapped_io, MLFlashpointCheckpointIO)
        assert wrapped_io.fallback_checkpoint_io is inner_checkpoint_io
        assert wrapped_io.async_save is False

    def test_mlflashpoint_enabled_with_multiple_callbacks(
        self, mocker, mock_ckpt_obj_manager, mock_replication_manager
    ):
        """Tests that wrapping occurs if the MLF callback is one of many."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [
            mocker.MagicMock(),
            mocker.MagicMock(spec=MLFlashpointCheckpointCallback),
            mocker.MagicMock(),
        ]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        original_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        trainer.strategy.checkpoint_io = original_checkpoint_io
        base_container = "/test_base_container"

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            replication_manager=mock_replication_manager,
            async_save=True,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
        )

        # Then
        # Wrapping should have occurred
        assert isinstance(trainer.strategy.checkpoint_io, MLFlashpointCheckpointIO)
        assert trainer.strategy.checkpoint_io.fallback_checkpoint_io is original_checkpoint_io

    def test_invalid_config_with_mlf_async_wrapper_and_async_save_false(
        self, mocker, mock_ckpt_obj_manager, mock_replication_manager
    ):
        """Tests that a ValueError is raised for an invalid configuration."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        base_container = "/test_base_container"

        # Create an already-wrapped MLFlashpointCheckpointIO
        mlf_io = mocker.MagicMock(spec=MLFlashpointCheckpointIO)
        original_async_wrapped_mlf_io = MLFlashpointAsyncFinalizableCheckpointIO(mlf_io)
        trainer.strategy.checkpoint_io = original_async_wrapped_mlf_io

        # When/Then
        with pytest.raises(ValueError, match="invalid configuration"):
            wrap_trainer_checkpoint_io_with_mlflashpoint(
                trainer,
                base_container,
                mock_ckpt_obj_manager,
                replication_manager=mock_replication_manager,
                async_save=False,
                checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
            )

    @pytest.mark.parametrize(
        "buffer_size_kwarg, expected_buffer_size",
        [
            ({}, DEFAULT_INITIAL_BUFFER_SIZE_BYTES),
            ({"initial_write_buffer_size_bytes": 12345}, 12345),
        ],
    )
    def test_initial_save_buffer_size_forwarding(
        self, mocker, mock_ckpt_obj_manager, mock_replication_manager, buffer_size_kwarg, expected_buffer_size
    ):
        """Tests that the initial_save_buffer_size_bytes is forwarded correctly."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        original_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        trainer.strategy.checkpoint_io = original_checkpoint_io
        base_container = "/test_base_container"

        # Patch the saver to check the arguments passed to it
        mock_saver = mocker.patch(
            "ml_flashpoint.adapter.nemo.wrapper_util.DefaultMLFlashpointCheckpointSaver",
        )

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            mock_replication_manager,
            async_save=True,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
            **buffer_size_kwarg,
        )

        # Then
        assert isinstance(trainer.strategy.checkpoint_io, MLFlashpointCheckpointIO)
        # Verify that the saver was initialized with the correct buffer size
        mock_saver.assert_called_once()
        _, kwargs = mock_saver.call_args
        assert kwargs["initial_buffer_size_bytes"] == expected_buffer_size

    @pytest.mark.parametrize(
        "thread_count_kwarg, expected_thread_count",
        [
            ({}, 1),
            ({"write_thread_count": 4}, 4),
        ],
    )
    def test_write_thread_count_forwarding(
        self, mocker, mock_ckpt_obj_manager, mock_replication_manager, thread_count_kwarg, expected_thread_count
    ):
        """Tests that the write_thread_count is forwarded correctly."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        original_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        trainer.strategy.checkpoint_io = original_checkpoint_io
        base_container = "/test_base_container"

        # Spy on the MemoryStorageWriter's __init__ method
        spy_memory_storage_writer_init = mocker.spy(MemoryStorageWriter, "__init__")

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            mock_replication_manager,
            async_save=True,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
            **thread_count_kwarg,
        )

        # Then
        # Verify that MemoryStorageWriter was initialized with the correct thread count
        spy_memory_storage_writer_init.assert_called_once()
        _, kwargs = spy_memory_storage_writer_init.call_args
        assert kwargs["thread_count"] == expected_thread_count

    @pytest.mark.parametrize("use_cached_ckpt_structure", [True, False])
    def test_cached_ckpt_structure_forwarding(
        self, mocker, mock_ckpt_obj_manager, mock_replication_manager, use_cached_ckpt_structure
    ):
        """Tests that use_cached_ckpt_structure is forwarded correctly."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        trainer.strategy.checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        base_container = "/test_base_container"

        # Mock the SaveStrategy to check initialization arguments
        mock_save_strategy_cls = mocker.patch(
            "ml_flashpoint.adapter.nemo.wrapper_util.MLFlashpointMegatronAsyncSaveStrategy"
        )

        # Mock dependencies
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.ReplicationManager")
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.MemoryStorageWriter")
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.DefaultMLFlashpointCheckpointSaver")
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.torch_mp.get_context")
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.MLFlashpointMegatronLoadStrategy")

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            mock_replication_manager,
            async_save=True,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
            use_cached_ckpt_structure=use_cached_ckpt_structure,
        )

        # Then
        mock_save_strategy_cls.assert_called_once()
        _, kwargs = mock_save_strategy_cls.call_args
        assert kwargs["use_cached_ckpt_structure"] == use_cached_ckpt_structure

    def test_spawn_context_used_for_mp_manager(self, mocker, mock_ckpt_obj_manager, mock_replication_manager):
        """Tests that torch_mp.get_context('spawn').Manager() is correctly instantiated and passed."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        original_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        trainer.strategy.checkpoint_io = original_checkpoint_io
        base_container = "/test_base_container"

        mock_get_context = mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.torch_mp.get_context")

        mock_ctx = mock_get_context.return_value  # The mocked context object
        mock_manager_instance = mock_ctx.Manager.return_value  # The mocked manager instance

        spy_memory_storage_writer_init = mocker.spy(MemoryStorageWriter, "__init__")

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            mock_replication_manager,
            async_save=True,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
        )

        # Then
        # Verify get_context was called explicitly with 'spawn'
        mock_get_context.assert_called_once_with("spawn")

        # Verify Manager() was called on the correct spawn context
        mock_ctx.Manager.assert_called_once()

        # Verify the exact Manager instance was passed to MemoryStorageWriter
        spy_memory_storage_writer_init.assert_called_once()
        _, kwargs = spy_memory_storage_writer_init.call_args
        assert kwargs["mp_manager"] is mock_manager_instance

    @pytest.mark.parametrize("always_save_context, expected_value", [(True, True), (False, False)])
    def test_always_save_context_forwarding(
        self, mocker, mock_ckpt_obj_manager, mock_replication_manager, always_save_context, expected_value
    ):
        """Tests that the always_save_context is forwarded correctly."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        original_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        trainer.strategy.checkpoint_io = original_checkpoint_io
        base_container = "/test_base_container"

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            mock_replication_manager,
            async_save=True,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
            always_save_context=always_save_context,
        )

        # Then
        assert isinstance(trainer.strategy.checkpoint_io, MLFlashpointCheckpointIO)
        assert trainer.strategy.checkpoint_io.always_save_context == expected_value

    @pytest.mark.parametrize("use_optimized_save", [True, False])
    def test_use_optimized_save_flag_passed_to_saver(
        self, mocker, mock_ckpt_obj_manager, mock_replication_manager, use_optimized_save
    ):
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        original_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        trainer.strategy.checkpoint_io = original_checkpoint_io

        # Patch DefaultMLFlashpointCheckpointSaver to check args
        mock_saver_cls = mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.DefaultMLFlashpointCheckpointSaver")

        flashpoint_base_container = "/test_base_container"
        async_save = True
        default_auto_resume = nl.AutoResume()
        mocker.patch("ml_flashpoint.adapter.nemo.wrapper_util.ReplicationManager")

        # When
        wrap_trainer_and_auto_resume_with_mlflashpoint(
            trainer, flashpoint_base_container, async_save, default_auto_resume, use_optimized_save=use_optimized_save
        )

        # Then
        mock_saver_cls.assert_called_once()
        _, kwargs = mock_saver_cls.call_args
        assert kwargs["use_optimized_save"] == use_optimized_save

    def test_checkpoint_loader_passed_to_load_strategy(self, mocker, mock_ckpt_obj_manager, mock_replication_manager):
        """Tests that the provided checkpoint_loader is passed to the load strategy."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        original_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        trainer.strategy.checkpoint_io = original_checkpoint_io
        base_container = "/test_base_container"

        mock_loader = mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader)

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            mock_replication_manager,
            async_save=True,
            checkpoint_loader=mock_loader,
        )

        # Then
        assert isinstance(trainer.strategy.checkpoint_io, MLFlashpointCheckpointIO)
        # Verify that the load strategy uses the passed checkpoint loader
        assert trainer.strategy.checkpoint_io.load_strategy.checkpoint_loader is mock_loader

    @pytest.mark.parametrize("use_optimized_save", [True, False])
    def test_use_optimized_save_flag_passed_to_wrap_trainer_checkpoint_io_with_mlflashpoint(
        self, mocker, mock_ckpt_obj_manager, mock_replication_manager, use_optimized_save
    ):
        """Tests that the use_optimized_save flag is forwarded correctly."""
        # Given
        trainer = mocker.MagicMock(spec=nl_trainer.Trainer)
        trainer.callbacks = [mocker.MagicMock(spec=MLFlashpointCheckpointCallback)]
        trainer.strategy = mocker.MagicMock(spec=nl_strategies.MegatronStrategy)
        original_checkpoint_io = mocker.MagicMock(spec=MegatronCheckpointIO)
        trainer.strategy.checkpoint_io = original_checkpoint_io
        base_container = "/test_base_container"

        # Patch the saver to check the arguments passed to it
        mock_saver = mocker.patch(
            "ml_flashpoint.adapter.nemo.wrapper_util.DefaultMLFlashpointCheckpointSaver",
        )

        # When
        wrap_trainer_checkpoint_io_with_mlflashpoint(
            trainer,
            base_container,
            mock_ckpt_obj_manager,
            mock_replication_manager,
            async_save=True,
            checkpoint_loader=mocker.MagicMock(spec=DefaultMLFlashpointCheckpointLoader),
            use_optimized_save=use_optimized_save,
        )

        # Then
        assert isinstance(trainer.strategy.checkpoint_io, MLFlashpointCheckpointIO)
        # Verify that the saver was initialized with the correct flag
        mock_saver.assert_called_once()
        _, kwargs = mock_saver.call_args
        assert kwargs["use_optimized_save"] == use_optimized_save
