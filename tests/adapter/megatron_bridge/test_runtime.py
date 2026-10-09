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

from ml_flashpoint.adapter.megatron_bridge import runtime as runtime_module
from ml_flashpoint.adapter.megatron_bridge.config import MLFlashpointBridgeConfig
from ml_flashpoint.adapter.megatron_bridge.runtime import (
    NUM_OF_BUFFERS_PER_OBJECT,
    MLFlashpointBridgeRuntime,
    get_runtime,
    shutdown_runtime,
)


@pytest.fixture
def config(tmp_path) -> MLFlashpointBridgeConfig:
    return MLFlashpointBridgeConfig(base_container=str(tmp_path / "mlf"), write_thread_count=2)


@pytest.fixture
def distributed(mocker):
    """Makes the runtime believe it is rank 0 of an initialized process group."""
    mocker.patch.object(runtime_module.dist, "is_available", return_value=True)
    mocker.patch.object(runtime_module.dist, "is_initialized", return_value=True)
    mocker.patch.object(runtime_module.dist, "get_rank", return_value=0)
    mocker.patch.object(runtime_module.dist, "get_node_local_rank", return_value=0)
    mocker.patch.object(runtime_module.dist, "get_world_size", return_value=1)
    mocker.patch.object(runtime_module.dist, "barrier")


@pytest.fixture
def collaborators(mocker):
    """Replaces everything the runtime constructs with doubles."""
    return {
        "object_manager": mocker.patch.object(runtime_module, "CheckpointObjectManager"),
        "replication_manager": mocker.patch.object(runtime_module, "ReplicationManager"),
        "saver": mocker.patch.object(runtime_module, "DefaultMLFlashpointCheckpointSaver"),
        "loader": mocker.patch.object(runtime_module, "DefaultMLFlashpointCheckpointLoader"),
        "storage_writer": mocker.patch.object(runtime_module, "MemoryStorageWriter"),
        "save_strategy": mocker.patch.object(runtime_module, "MLFlashpointMegatronAsyncSaveStrategy"),
        "load_strategy": mocker.patch.object(runtime_module, "MLFlashpointMegatronLoadStrategy"),
        "parallel_save": mocker.patch.object(runtime_module, "FullyParallelSaveStrategyWrapper"),
        "parallel_load": mocker.patch.object(runtime_module, "FullyParallelLoadStrategyWrapper"),
        "queue": mocker.patch.object(runtime_module, "AsyncCallsQueue"),
        "mp": mocker.patch.object(runtime_module, "torch_mp"),
    }


@pytest.fixture(autouse=True)
def _clean_module_runtime():
    runtime_module._RUNTIME = None
    yield
    runtime_module._RUNTIME = None


class TestConstruction:
    def test_requires_an_initialized_process_group(self, config, mocker, collaborators):
        # Given
        mocker.patch.object(runtime_module.dist, "is_initialized", return_value=False)

        # When/Then
        with pytest.raises(RuntimeError, match="torch.distributed"):
            MLFlashpointBridgeRuntime(config)

    def test_sizes_the_buffer_pool_from_the_thread_count(self, config, distributed, collaborators, mocker):
        # Given
        pool_config = mocker.patch.object(runtime_module, "BufferPoolConfig")

        # When
        MLFlashpointBridgeRuntime(config)

        # Then
        assert_that(pool_config.call_args.kwargs["num_buffers"]).is_equal_to(
            config.write_thread_count * NUM_OF_BUFFERS_PER_OBJECT
        )

    def test_initializes_replication(self, config, distributed, collaborators):
        # Given/When
        MLFlashpointBridgeRuntime(config)

        # Then
        collaborators["replication_manager"].return_value.initialize.assert_called_once()

    def test_wraps_strategies_for_fully_parallel_saving(self, config, distributed, collaborators):
        # Given/When
        MLFlashpointBridgeRuntime(config)

        # Then
        collaborators["parallel_save"].assert_called_once()
        collaborators["parallel_load"].assert_called_once()

    def test_skips_the_parallel_wrapper_when_disabled(self, tmp_path, distributed, collaborators):
        # Given
        config = MLFlashpointBridgeConfig(base_container=str(tmp_path / "mlf"), use_fully_parallel_wrapper=False)

        # When
        runtime = MLFlashpointBridgeRuntime(config)

        # Then
        collaborators["parallel_save"].assert_not_called()
        assert_that(runtime.save_strategy).is_same_as(collaborators["save_strategy"].return_value)

    def test_uses_a_persistent_async_queue(self, config, distributed, collaborators):
        # Given/When
        MLFlashpointBridgeRuntime(config)

        # Then
        collaborators["queue"].assert_called_once_with(persistent=True)

    def test_exposes_the_base_container(self, config, distributed, collaborators):
        # Given/When
        runtime = MLFlashpointBridgeRuntime(config)

        # Then
        assert_that(str(runtime.base_container)).is_equal_to(config.base_container)
        assert_that(runtime.config).is_same_as(config)


class TestAsyncQueue:
    def test_schedule_forwards_to_the_queue(self, config, distributed, collaborators):
        # Given
        runtime = MLFlashpointBridgeRuntime(config)
        queue = collaborators["queue"].return_value
        queue.schedule_async_request.return_value = 7

        # When
        result = runtime.schedule("request")

        # Then
        assert_that(result).is_equal_to(7)
        queue.schedule_async_request.assert_called_once_with("request")

    def test_maybe_finalize_short_circuits_when_idle(self, config, distributed, collaborators):
        # Given
        runtime = MLFlashpointBridgeRuntime(config)
        queue = collaborators["queue"].return_value
        queue.get_num_unfinalized_calls.return_value = 0

        # When
        result = runtime.maybe_finalize()

        # Then
        assert_that(result).is_false()
        queue.maybe_finalize_async_calls.assert_not_called()

    def test_maybe_finalize_reports_completions(self, config, distributed, collaborators):
        # Given
        runtime = MLFlashpointBridgeRuntime(config)
        queue = collaborators["queue"].return_value
        queue.get_num_unfinalized_calls.return_value = 2
        queue.maybe_finalize_async_calls.return_value = [0, 1]

        # When
        result = runtime.maybe_finalize(blocking=True)

        # Then
        assert_that(result).is_true()
        queue.maybe_finalize_async_calls.assert_called_once_with(True)

    def test_num_unfinalized_calls_is_forwarded(self, config, distributed, collaborators):
        # Given
        runtime = MLFlashpointBridgeRuntime(config)
        collaborators["queue"].return_value.get_num_unfinalized_calls.return_value = 3

        # When/Then
        assert_that(runtime.num_unfinalized_calls()).is_equal_to(3)


class TestShutdown:
    def test_releases_replication_buffers_and_queue(self, config, distributed, collaborators):
        # Given
        runtime = MLFlashpointBridgeRuntime(config)

        # When
        runtime.shutdown()

        # Then
        collaborators["replication_manager"].return_value.shutdown.assert_called_once()
        collaborators["object_manager"].return_value.delete_container.assert_called_once()
        collaborators["queue"].return_value.close.assert_called_once()

    def test_can_keep_the_container(self, config, distributed, collaborators):
        # Given
        runtime = MLFlashpointBridgeRuntime(config)

        # When
        runtime.shutdown(remove_checkpoints=False)

        # Then
        collaborators["object_manager"].return_value.delete_container.assert_not_called()

    def test_is_idempotent(self, config, distributed, collaborators):
        # Given
        runtime = MLFlashpointBridgeRuntime(config)

        # When
        runtime.shutdown()
        runtime.shutdown()

        # Then
        assert_that(collaborators["queue"].return_value.close.call_count).is_equal_to(1)

    def test_replication_failure_does_not_stop_teardown(self, config, distributed, collaborators):
        # Given
        collaborators["replication_manager"].return_value.shutdown.side_effect = RuntimeError("socket closed")
        runtime = MLFlashpointBridgeRuntime(config)

        # When
        runtime.shutdown()

        # Then
        collaborators["queue"].return_value.close.assert_called_once()

    def test_container_deletion_failure_does_not_stop_teardown(self, config, distributed, collaborators):
        # Given
        collaborators["object_manager"].return_value.delete_container.side_effect = OSError("busy")
        runtime = MLFlashpointBridgeRuntime(config)

        # When
        runtime.shutdown()

        # Then
        collaborators["queue"].return_value.close.assert_called_once()

    def test_finalize_after_shutdown_is_a_noop(self, config, distributed, collaborators):
        # Given
        runtime = MLFlashpointBridgeRuntime(config)
        runtime.shutdown()
        collaborators["queue"].return_value.get_num_unfinalized_calls.return_value = 5

        # When
        result = runtime.maybe_finalize()

        # Then
        assert_that(result).is_false()

    def test_only_local_rank_zero_deletes_the_container(self, config, distributed, collaborators, mocker):
        # Given
        mocker.patch.object(runtime_module.dist, "get_node_local_rank", return_value=1)
        runtime = MLFlashpointBridgeRuntime(config)

        # When
        runtime.shutdown()

        # Then
        collaborators["object_manager"].return_value.delete_container.assert_not_called()


class TestModuleRuntime:
    def test_get_runtime_builds_once(self, config, distributed, collaborators, mocker):
        # Given
        build = mocker.spy(runtime_module, "MLFlashpointBridgeRuntime")

        # When
        first = get_runtime(config)
        second = get_runtime(config)

        # Then
        assert_that(second).is_same_as(first)
        assert_that(build.call_count).is_equal_to(1)

    def test_shutdown_runtime_clears_the_module_state(self, config, distributed, collaborators):
        # Given
        get_runtime(config)

        # When
        shutdown_runtime()

        # Then
        assert_that(runtime_module._RUNTIME).is_none()

    def test_shutdown_runtime_without_a_runtime_is_a_noop(self):
        # Given/When
        shutdown_runtime()

        # Then
        assert_that(runtime_module._RUNTIME).is_none()

    def test_shutdown_runtime_clears_state_even_on_failure(self, config, distributed, collaborators, mocker):
        # Given
        runtime = get_runtime(config)
        mocker.patch.object(runtime, "shutdown", side_effect=RuntimeError("teardown failed"))

        # When/Then
        with pytest.raises(RuntimeError):
            shutdown_runtime()
        assert_that(runtime_module._RUNTIME).is_none()
