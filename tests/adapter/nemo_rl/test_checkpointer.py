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
from assertpy import assert_that

from ml_flashpoint.adapter.nemo_rl import checkpointer as checkpointer_module
from ml_flashpoint.adapter.nemo_rl.checkpointer import (
    MLFlashpointNeMoRLCheckpointer,
    _as_module_list,
)


@pytest.fixture
def manager(mocker):
    """Replaces the Bridge manager so no ML Flashpoint runtime is built."""
    instance = mocker.MagicMock()
    mocker.patch.object(checkpointer_module, "MLFlashpointBridgeCheckpointManager", return_value=instance)
    return instance


@pytest.fixture
def checkpoint_config():
    return type("CheckpointConfig", (), {"ckpt_format": "torch_dist"})()


@pytest.fixture
def state():
    train_state = type("TrainState", (), {"step": 12, "floating_point_operations_so_far": 555})()
    return type("GlobalState", (), {"train_state": train_state})()


@pytest.fixture
def checkpointer(checkpoint_config, manager) -> MLFlashpointNeMoRLCheckpointer:
    return MLFlashpointNeMoRLCheckpointer(checkpoint_config)


class TestSave:
    def test_marks_the_checkpoint_non_persistent(self, checkpointer, manager, state):
        # Given
        model = object()

        # When
        checkpointer.save(state=state, model=model)

        # Then
        ctx = manager.save.call_args.args[0]
        assert_that(ctx.non_persistent_ckpt).is_true()
        assert_that(ctx.state).is_same_as(state)
        assert_that(ctx.model).is_equal_to([model])

    def test_defaults_flops_from_the_train_state(self, checkpointer, manager, state):
        # Given/When
        checkpointer.save(state=state, model=object())

        # Then
        assert_that(manager.save.call_args.args[0].num_floating_point_operations_so_far).is_equal_to(555)

    def test_explicit_flops_win(self, checkpointer, manager, state):
        # Given/When
        checkpointer.save(state=state, model=object(), num_floating_point_operations_so_far=1)

        # Then
        assert_that(manager.save.call_args.args[0].num_floating_point_operations_so_far).is_equal_to(1)

    def test_passes_optimizer_and_scheduler_through(self, checkpointer, manager, state, mocker):
        # Given
        optimizer = mocker.MagicMock()
        scheduler = mocker.MagicMock()

        # When
        checkpointer.save(state=state, model=object(), optimizer=optimizer, opt_param_scheduler=scheduler)

        # Then
        ctx = manager.save.call_args.args[0]
        assert_that(ctx.optimizer).is_same_as(optimizer)
        assert_that(ctx.opt_param_scheduler).is_same_as(scheduler)

    def test_no_callback_manager_is_passed(self, checkpointer, manager, state):
        # Given/When
        checkpointer.save(state=state, model=object())

        # Then
        assert_that(manager.save.call_args.kwargs["callback_manager"]).is_none()


class TestLoad:
    def test_returns_none_when_nothing_recoverable(self, checkpointer, manager, state):
        # Given
        manager._find_ml_flashpoint_checkpoint.return_value = None

        # When
        result = checkpointer.load(state=state, model=object())

        # Then
        assert_that(result).is_none()
        manager._load_ml_flashpoint.assert_not_called()

    def test_loads_the_discovered_container(self, checkpointer, manager, state):
        # Given
        manager._find_ml_flashpoint_checkpoint.return_value = "container"
        manager._load_ml_flashpoint.return_value = (12, 555)

        # When
        result = checkpointer.load(state=state, model=object())

        # Then
        assert_that(result).is_equal_to((12, 555))
        assert_that(manager._load_ml_flashpoint.call_args.args[1]).is_equal_to("container")

    def test_never_falls_back_to_the_durable_path(self, checkpointer, manager, state):
        # Given: NeMo RL owns the durable resume decision.
        manager._find_ml_flashpoint_checkpoint.return_value = "container"
        manager._load_ml_flashpoint.return_value = None

        # When
        result = checkpointer.load(state=state, model=object())

        # Then
        assert_that(result).is_none()
        manager.load.assert_not_called()

    def test_forwards_strictness(self, checkpointer, manager, state):
        # Given
        manager._find_ml_flashpoint_checkpoint.return_value = "container"

        # When
        checkpointer.load(state=state, model=object(), strict=False)

        # Then
        assert_that(manager._load_ml_flashpoint.call_args.args[0].strict).is_false()


class TestLifecycle:
    def test_enabled_follows_the_manager(self, checkpointer, manager):
        # Given
        manager.enabled = False

        # When/Then
        assert_that(checkpointer.enabled).is_false()

    def test_finalize_without_a_runtime_is_a_noop(self, checkpointer, manager):
        # Given
        manager.runtime = None

        # When
        checkpointer.finalize()

        # Then no exception escapes.

    def test_finalize_drains_the_runtime(self, checkpointer, manager, mocker):
        # Given
        runtime = mocker.MagicMock()
        manager.runtime = runtime

        # When
        checkpointer.finalize(blocking=True)

        # Then
        runtime.maybe_finalize.assert_called_once_with(blocking=True)

    def test_shutdown_delegates_to_the_manager(self, checkpointer, manager):
        # Given/When
        checkpointer.shutdown()

        # Then
        manager.shutdown.assert_called_once()


class TestAsModuleList:
    def test_wraps_a_single_module(self):
        # Given
        module = object()

        # When/Then
        assert_that(_as_module_list(module)).is_equal_to([module])

    def test_passes_a_list_through(self):
        # Given
        modules = [object(), object()]

        # When/Then
        assert_that(_as_module_list(modules)).is_equal_to(modules)

    def test_normalizes_a_tuple(self):
        # Given
        modules = (object(),)

        # When/Then
        assert_that(_as_module_list(modules)).is_equal_to(list(modules))
