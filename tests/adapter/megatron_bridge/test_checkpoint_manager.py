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
from typing import Any, Optional

import pytest
from assertpy import assert_that

from ml_flashpoint.adapter.megatron_bridge import checkpoint_manager as manager_module
from ml_flashpoint.adapter.megatron_bridge.checkpoint_manager import (
    MLFlashpointBridgeCheckpointManager,
    init_checkpointing_context_safely,
)
from ml_flashpoint.adapter.megatron_bridge.config import MLFlashpointBridgeConfig
from ml_flashpoint.adapter.megatron_bridge.local_checkpoint_index import (
    NO_CHECKPOINT,
    MLFlashpointLocalCheckpointIndex,
)
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId


@dataclasses.dataclass
class FakeCheckpointConfig:
    """The subset of Megatron Bridge's ``CheckpointConfig`` the adapter reads."""

    save: str = "/durable/checkpoints"
    ckpt_format: str = "torch_dist"
    non_persistent_ckpt_type: str = "local"
    non_persistent_save_interval: int = 10
    save_interval: int = 100
    custom_manager_class: Optional[str] = None


@dataclasses.dataclass
class FakeTrainState:
    step: int = 0
    floating_point_operations_so_far: int = 0


@dataclasses.dataclass
class FakeGlobalState:
    cfg: Any = None
    train_state: FakeTrainState = dataclasses.field(default_factory=FakeTrainState)


@dataclasses.dataclass
class FakeSaveContext:
    state: FakeGlobalState
    model: list
    optimizer: Any = None
    opt_param_scheduler: Any = None
    num_floating_point_operations_so_far: int = 0
    train_data_iterator: Any = None
    non_persistent_ckpt: bool = False
    pg_collection: Any = None
    module_name: Optional[str] = None


@dataclasses.dataclass
class FakeLoadContext:
    state: FakeGlobalState
    model: list
    optimizer: Any = None
    opt_param_scheduler: Any = None
    strict: bool = True
    skip_load_to_model_and_opt: bool = False
    pg_collection: Any = None
    module_name: Optional[str] = None


@pytest.fixture
def checkpoint_config() -> FakeCheckpointConfig:
    return FakeCheckpointConfig()


@pytest.fixture
def mlf_config(tmp_path) -> MLFlashpointBridgeConfig:
    return MLFlashpointBridgeConfig(base_container=str(tmp_path / "mlf"))


@pytest.fixture
def runtime(mocker, mlf_config):
    """A runtime double whose base container matches the config under test."""
    fake = mocker.MagicMock()
    fake.base_container = CheckpointContainerId(mlf_config.base_container)
    fake.config = mlf_config
    return fake


@pytest.fixture
def global_state(checkpoint_config) -> FakeGlobalState:
    cfg = mocker_namespace(checkpoint=checkpoint_config)
    return FakeGlobalState(cfg=cfg, train_state=FakeTrainState(step=40))


def mocker_namespace(**kwargs):
    """Builds a lightweight attribute container for nested config objects."""
    return type("Namespace", (), kwargs)()


@pytest.fixture
def manager(checkpoint_config, mlf_config, runtime, mocker) -> MLFlashpointBridgeCheckpointManager:
    mocker.patch.object(manager_module, "init_checkpointing_context", return_value={})
    return MLFlashpointBridgeCheckpointManager(
        checkpoint_config=checkpoint_config,
        mlf_config=mlf_config,
        runtime=runtime,
    )


class TestEnablement:
    def test_enabled_by_default(self, manager):
        # Given/When/Then
        assert_that(manager.enabled).is_true()

    def test_disabled_when_config_disabled(self, checkpoint_config, runtime, mocker, tmp_path):
        # Given
        mocker.patch.object(manager_module, "init_checkpointing_context", return_value={})
        config = MLFlashpointBridgeConfig(enabled=False, base_container=str(tmp_path / "mlf"))

        # When
        manager = MLFlashpointBridgeCheckpointManager(checkpoint_config, mlf_config=config, runtime=runtime)

        # Then
        assert_that(manager.enabled).is_false()

    @pytest.mark.parametrize("ckpt_format", ["torch", "fsdp_dtensor"])
    def test_disabled_for_unsupported_checkpoint_format(self, ckpt_format, mlf_config, runtime, mocker):
        # Given
        mocker.patch.object(manager_module, "init_checkpointing_context", return_value={})
        checkpoint_config = FakeCheckpointConfig(ckpt_format=ckpt_format)

        # When
        manager = MLFlashpointBridgeCheckpointManager(checkpoint_config, mlf_config=mlf_config, runtime=runtime)

        # Then
        assert_that(manager.enabled).is_false()

    def test_installs_local_index_into_context(self, manager):
        # Given/When
        context = manager.checkpointing_context

        # Then
        assert_that(context).contains_key("local_checkpoint_manager")
        assert_that(context["local_checkpoint_manager"]).is_instance_of(MLFlashpointLocalCheckpointIndex)

    def test_disabled_manager_does_not_install_local_index(self, checkpoint_config, runtime, mocker, tmp_path):
        # Given
        mocker.patch.object(manager_module, "init_checkpointing_context", return_value={})
        config = MLFlashpointBridgeConfig(enabled=False, base_container=str(tmp_path / "mlf"))

        # When
        manager = MLFlashpointBridgeCheckpointManager(checkpoint_config, mlf_config=config, runtime=runtime)

        # Then
        assert_that(manager.checkpointing_context).does_not_contain_key("local_checkpoint_manager")


class TestSaveRouting:
    def test_persistent_save_is_delegated(self, manager, global_state, mocker):
        # Given
        delegate = mocker.patch.object(manager_module, "save_checkpoint")
        mlf_save = mocker.patch.object(manager_module, "save_local_aware_megatron_checkpoint")
        ctx = FakeSaveContext(state=global_state, model=[object()], non_persistent_ckpt=False)

        # When
        manager.save(ctx, callback_manager=None)

        # Then
        delegate.assert_called_once()
        mlf_save.assert_not_called()

    def test_non_persistent_save_goes_to_ml_flashpoint(self, manager, global_state, mocker):
        # Given
        delegate = mocker.patch.object(manager_module, "save_checkpoint")
        mocker.patch.object(manager_module.bridge_state, "build_save_state_dict", return_value={"model": {}})
        mlf_save = mocker.patch.object(
            manager_module, "save_local_aware_megatron_checkpoint", return_value="async-request"
        )
        ctx = FakeSaveContext(state=global_state, model=[object()], non_persistent_ckpt=True)

        # When
        manager.save(ctx, callback_manager=None)

        # Then
        delegate.assert_not_called()
        mlf_save.assert_called_once()
        assert_that(mlf_save.call_args.kwargs["checkpoint_dir"]).ends_with("step-40_ckpt")
        manager.runtime.schedule.assert_called_once_with("async-request")

    def test_synchronous_save_schedules_nothing(self, checkpoint_config, runtime, global_state, mocker, tmp_path):
        # Given
        mocker.patch.object(manager_module, "init_checkpointing_context", return_value={})
        config = MLFlashpointBridgeConfig(async_save=False, base_container=str(tmp_path / "mlf"))
        manager = MLFlashpointBridgeCheckpointManager(checkpoint_config, mlf_config=config, runtime=runtime)
        mocker.patch.object(manager_module.bridge_state, "build_save_state_dict", return_value={})
        mocker.patch.object(manager_module, "save_local_aware_megatron_checkpoint", return_value=None)
        ctx = FakeSaveContext(state=global_state, model=[object()], non_persistent_ckpt=True)

        # When
        manager.save(ctx, callback_manager=None)

        # Then
        runtime.schedule.assert_not_called()

    def test_ml_flashpoint_failure_does_not_propagate(self, manager, global_state, mocker):
        # Given
        mocker.patch.object(
            manager_module.bridge_state, "build_save_state_dict", side_effect=RuntimeError("out of buffers")
        )
        ctx = FakeSaveContext(state=global_state, model=[object()], non_persistent_ckpt=True)

        # When
        manager.save(ctx, callback_manager=None)

        # Then no exception escapes; the training loop keeps going.

    def test_disabled_manager_delegates_non_persistent_save(
        self, checkpoint_config, runtime, global_state, mocker, tmp_path
    ):
        # Given a run that opted out of ML Flashpoint but kept the local cadence.
        mocker.patch.object(manager_module, "init_checkpointing_context", return_value={})
        config = MLFlashpointBridgeConfig(enabled=False, base_container=str(tmp_path / "mlf"))
        manager = MLFlashpointBridgeCheckpointManager(checkpoint_config, mlf_config=config, runtime=runtime)
        ctx = FakeSaveContext(state=global_state, model=[object()], non_persistent_ckpt=True)

        # When/Then: Bridge cannot service a local checkpoint ML Flashpoint owns.
        with pytest.raises(RuntimeError, match="non_persistent_ckpt_type='local'"):
            manager.save(ctx, callback_manager=None)

    def test_disabled_manager_delegates_global_non_persistent_save(self, runtime, mocker, tmp_path):
        # Given
        mocker.patch.object(manager_module, "init_checkpointing_context", return_value={})
        delegate = mocker.patch.object(manager_module, "save_checkpoint")
        checkpoint_config = FakeCheckpointConfig(non_persistent_ckpt_type="global")
        config = MLFlashpointBridgeConfig(enabled=False, base_container=str(tmp_path / "mlf"))
        manager = MLFlashpointBridgeCheckpointManager(checkpoint_config, mlf_config=config, runtime=runtime)
        state = FakeGlobalState(cfg=mocker_namespace(checkpoint=checkpoint_config), train_state=FakeTrainState(1))
        ctx = FakeSaveContext(state=state, model=[object()], non_persistent_ckpt=True)

        # When
        manager.save(ctx, callback_manager=None)

        # Then
        delegate.assert_called_once()

    def test_save_invalidates_cached_resume_decision(self, manager, global_state, mocker):
        # Given
        mocker.patch.object(manager_module.bridge_state, "build_save_state_dict", return_value={})
        mocker.patch.object(manager_module, "save_local_aware_megatron_checkpoint", return_value=None)
        index = manager.checkpointing_context["local_checkpoint_manager"]
        invalidate = mocker.spy(index, "invalidate")
        ctx = FakeSaveContext(state=global_state, model=[object()], non_persistent_ckpt=True)

        # When
        manager.save(ctx, callback_manager=None)

        # Then
        assert_that(invalidate.call_count).is_equal_to(1)


class TestLoadRouting:
    def test_prefers_ml_flashpoint_checkpoint(self, manager, global_state, mocker):
        # Given
        container = CheckpointContainerId(str(manager.runtime.base_container) + "/step-40_ckpt")
        mocker.patch.object(manager, "_find_ml_flashpoint_checkpoint", return_value=container)
        mocker.patch.object(manager, "_load_ml_flashpoint", return_value=(40, 1234))
        delegate = mocker.patch.object(manager_module, "load_checkpoint")
        ctx = FakeLoadContext(state=global_state, model=[object()])

        # When
        result = manager.load(ctx)

        # Then
        assert_that(result).is_equal_to((40, 1234))
        delegate.assert_not_called()

    def test_falls_back_when_no_ml_flashpoint_checkpoint(self, manager, global_state, mocker):
        # Given
        mocker.patch.object(manager, "_find_ml_flashpoint_checkpoint", return_value=None)
        delegate = mocker.patch.object(manager_module, "load_checkpoint", return_value=(7, 99))
        ctx = FakeLoadContext(state=global_state, model=[object()])

        # When
        result = manager.load(ctx)

        # Then
        assert_that(result).is_equal_to((7, 99))
        delegate.assert_called_once()

    def test_falls_back_when_ml_flashpoint_read_fails(self, manager, global_state, mocker):
        # Given
        container = CheckpointContainerId(str(manager.runtime.base_container) + "/step-40_ckpt")
        mocker.patch.object(manager, "_find_ml_flashpoint_checkpoint", return_value=container)
        mocker.patch.object(manager, "_load_ml_flashpoint", return_value=None)
        delegate = mocker.patch.object(manager_module, "load_checkpoint", return_value=(0, 0))
        ctx = FakeLoadContext(state=global_state, model=[object()])

        # When
        result = manager.load(ctx)

        # Then
        assert_that(result).is_equal_to((0, 0))
        delegate.assert_called_once()

    def test_fallback_hides_the_local_index_from_bridge(self, manager, global_state, mocker):
        # Given: Bridge's own local path cannot read an ML Flashpoint container.
        mocker.patch.object(manager, "_find_ml_flashpoint_checkpoint", return_value=None)
        mocker.patch.object(manager_module, "load_checkpoint", return_value=(0, 0))
        index = manager.checkpointing_context["local_checkpoint_manager"]
        ctx = FakeLoadContext(state=global_state, model=[object()])

        # When
        manager.load(ctx)

        # Then
        assert_that(index.find_latest()).is_equal_to(NO_CHECKPOINT)

    def test_read_failure_returns_none_for_fallback(self, manager, global_state, mocker):
        # Given
        container = CheckpointContainerId(str(manager.runtime.base_container) + "/step-40_ckpt")
        mocker.patch.object(
            manager_module.bridge_state, "build_load_state_dict", side_effect=RuntimeError("missing objects")
        )
        ctx = FakeLoadContext(state=global_state, model=[object()])

        # When
        result = manager._load_ml_flashpoint(ctx, container)

        # Then
        assert_that(result).is_none()

    def test_successful_read_applies_state(self, manager, global_state, mocker):
        # Given
        container = CheckpointContainerId(str(manager.runtime.base_container) + "/step-40_ckpt")
        mocker.patch.object(manager_module.bridge_state, "build_load_state_dict", return_value={})
        mocker.patch.object(manager_module.mcore_dist_checkpointing, "load", return_value={"model": {}})
        apply_state = mocker.patch.object(manager_module.bridge_state, "apply_loaded_state", return_value=(40, 5))
        ctx = FakeLoadContext(state=global_state, model=[object()])

        # When
        result = manager._load_ml_flashpoint(ctx, container)

        # Then
        assert_that(result).is_equal_to((40, 5))
        apply_state.assert_called_once()


class TestFinalizeAndShutdown:
    def test_finalizes_both_queues(self, manager, global_state, mocker):
        # Given
        bridge_finalize = mocker.patch.object(manager_module, "maybe_finalize_async_save")

        # When
        manager.finalize_async_saves(global_state, blocking=False)

        # Then
        manager.runtime.maybe_finalize.assert_called_once_with(blocking=False)
        bridge_finalize.assert_called_once()

    def test_ml_flashpoint_finalize_failure_still_finalizes_bridge(self, manager, global_state, mocker):
        # Given
        manager.runtime.maybe_finalize.side_effect = RuntimeError("worker died")
        bridge_finalize = mocker.patch.object(manager_module, "maybe_finalize_async_save")

        # When
        manager.finalize_async_saves(global_state, blocking=True)

        # Then
        bridge_finalize.assert_called_once()

    def test_terminate_shuts_the_runtime_down(self, manager, global_state, mocker):
        # Given
        mocker.patch.object(manager_module, "maybe_finalize_async_save")
        shutdown = mocker.patch("ml_flashpoint.adapter.megatron_bridge.runtime.shutdown_runtime")
        mocker.patch.object(manager_module.dist, "is_initialized", return_value=False)

        # When
        manager.finalize_async_saves(global_state, blocking=True, terminate=True)

        # Then
        shutdown.assert_called_once_with(remove_checkpoints=True)
        assert_that(manager.runtime).is_none()

    def test_shutdown_keeps_checkpoints_when_configured(self, checkpoint_config, runtime, mocker, tmp_path):
        # Given
        mocker.patch.object(manager_module, "init_checkpointing_context", return_value={})
        mocker.patch.object(manager_module.dist, "is_initialized", return_value=False)
        shutdown = mocker.patch("ml_flashpoint.adapter.megatron_bridge.runtime.shutdown_runtime")
        config = MLFlashpointBridgeConfig(base_container=str(tmp_path / "mlf"), keep_checkpoints_on_finalize=True)
        manager = MLFlashpointBridgeCheckpointManager(checkpoint_config, mlf_config=config, runtime=runtime)

        # When
        manager.shutdown()

        # Then
        shutdown.assert_called_once_with(remove_checkpoints=False)

    def test_shutdown_without_runtime_is_a_noop(self, checkpoint_config, mlf_config, mocker):
        # Given
        mocker.patch.object(manager_module, "init_checkpointing_context", return_value={})
        shutdown = mocker.patch("ml_flashpoint.adapter.megatron_bridge.runtime.shutdown_runtime")
        manager = MLFlashpointBridgeCheckpointManager(checkpoint_config, mlf_config=mlf_config)

        # When
        manager.shutdown()

        # Then
        shutdown.assert_not_called()

    def test_shutdown_clears_the_local_index_from_context(self, manager, mocker):
        # Given
        mocker.patch.object(manager_module.dist, "is_initialized", return_value=False)
        mocker.patch("ml_flashpoint.adapter.megatron_bridge.runtime.shutdown_runtime")
        assert_that(manager.checkpointing_context).contains_key("local_checkpoint_manager")

        # When
        manager.shutdown()

        # Then
        assert_that(manager._context).does_not_contain_key("local_checkpoint_manager")


class TestRuntimeBootstrap:
    def test_runtime_is_built_lazily(self, checkpoint_config, mlf_config, mocker):
        # Given
        mocker.patch.object(manager_module, "init_checkpointing_context", return_value={})
        built = mocker.MagicMock()
        built.base_container = CheckpointContainerId(mlf_config.base_container)
        get_runtime = mocker.patch.object(manager_module, "get_runtime", return_value=built)
        manager = MLFlashpointBridgeCheckpointManager(checkpoint_config, mlf_config=mlf_config)

        # When
        assert_that(manager.runtime).is_none()
        result = manager._ensure_runtime()

        # Then
        assert_that(result).is_same_as(built)
        get_runtime.assert_called_once_with(mlf_config)

    def test_runtime_failure_disables_ml_flashpoint(self, checkpoint_config, mlf_config, mocker):
        # Given
        mocker.patch.object(manager_module, "init_checkpointing_context", return_value={})
        mocker.patch.object(manager_module, "get_runtime", side_effect=RuntimeError("no process group"))
        manager = MLFlashpointBridgeCheckpointManager(checkpoint_config, mlf_config=mlf_config)

        # When
        result = manager._ensure_runtime()

        # Then
        assert_that(result).is_none()
        assert_that(manager.enabled).is_false()


class TestInitCheckpointingContextSafely:
    def test_passes_through(self, mocker, checkpoint_config):
        # Given
        mocker.patch.object(manager_module, "init_checkpointing_context", return_value={"a": 1})

        # When
        result = init_checkpointing_context_safely(checkpoint_config)

        # Then
        assert_that(result).is_equal_to({"a": 1})

    def test_missing_nvrx_yields_empty_context(self, mocker, checkpoint_config):
        # Given
        mocker.patch.object(
            manager_module,
            "init_checkpointing_context",
            side_effect=RuntimeError("nvidia_resiliency_ext is required"),
        )

        # When
        result = init_checkpointing_context_safely(checkpoint_config)

        # Then
        assert_that(result).is_equal_to({})
