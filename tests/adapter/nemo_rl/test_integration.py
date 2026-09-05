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

from ml_flashpoint.adapter.nemo_rl import integration as integration_module
from ml_flashpoint.adapter.nemo_rl.integration import (
    MODE_AUGMENT,
    MODE_REPLACE,
    get_checkpointer,
    install_from_env,
    install_into_worker,
    uninstall_from_worker,
)


class FakeWorker:
    """Stands in for NeMo RL's ``MegatronPolicyWorker``."""

    def __init__(self, checkpoint_config):
        self.mcore_state = type("State", (), {"cfg": type("Cfg", (), {"checkpoint": checkpoint_config})()})()
        self.model = object()
        self.optimizer = object()
        self.scheduler = object()
        self.durable_saves: list[tuple[str, str]] = []

    def save_checkpoint(self, weights_path, optimizer_path=None, **kwargs):
        self.durable_saves.append((weights_path, optimizer_path))
        return "durable-result"


@pytest.fixture
def checkpoint_config():
    return type("CheckpointConfig", (), {"ckpt_format": "torch_dist", "non_persistent_ckpt_type": "local"})()


@pytest.fixture
def worker(checkpoint_config) -> FakeWorker:
    return FakeWorker(checkpoint_config)


@pytest.fixture
def checkpointer(mocker):
    """Replaces the real checkpointer so no ML Flashpoint runtime is built."""
    instance = mocker.MagicMock()
    mocker.patch.object(integration_module, "MLFlashpointNeMoRLCheckpointer", return_value=instance)
    return instance


class TestInstall:
    def test_returns_checkpointer_and_records_it(self, worker, checkpointer):
        # Given/When
        result = install_into_worker(worker)

        # Then
        assert_that(result).is_same_as(checkpointer)
        assert_that(get_checkpointer(worker)).is_same_as(checkpointer)

    def test_installing_twice_is_a_noop(self, worker, checkpointer):
        # Given
        first = install_into_worker(worker)

        # When
        second = install_into_worker(worker)

        # Then
        assert_that(second).is_same_as(first)

    @pytest.mark.parametrize("mode", ["", "both", "AUGMENT"])
    def test_rejects_unknown_mode(self, worker, checkpointer, mode):
        # Given/When/Then
        with pytest.raises(ValueError, match="mode must be one of"):
            install_into_worker(worker, mode=mode)

    @pytest.mark.parametrize("interval", [0, -1])
    def test_rejects_non_positive_durable_interval(self, worker, checkpointer, interval):
        # Given/When/Then
        with pytest.raises(ValueError, match="durable_every_n_saves"):
            install_into_worker(worker, mode=MODE_REPLACE, durable_every_n_saves=interval)

    def test_rejects_object_that_is_not_a_megatron_worker(self, checkpointer):
        # Given
        not_a_worker = object()

        # When/Then
        with pytest.raises(AttributeError, match="mcore_state"):
            install_into_worker(not_a_worker)


class TestAugmentMode:
    def test_every_save_stays_durable(self, worker, checkpointer):
        # Given
        install_into_worker(worker, mode=MODE_AUGMENT)

        # When
        for step in range(3):
            worker.save_checkpoint(f"/durable/step_{step}", optimizer_path=f"/durable/step_{step}/optim")

        # Then
        assert_that(worker.durable_saves).is_length(3)
        assert_that(checkpointer.save.call_count).is_equal_to(3)

    def test_returns_the_wrapped_result(self, worker, checkpointer):
        # Given
        install_into_worker(worker, mode=MODE_AUGMENT)

        # When
        result = worker.save_checkpoint("/durable/step_0")

        # Then
        assert_that(result).is_equal_to("durable-result")

    def test_optimizer_is_skipped_when_no_optimizer_path(self, worker, checkpointer):
        # Given
        install_into_worker(worker, mode=MODE_AUGMENT)

        # When
        worker.save_checkpoint("/durable/step_0", optimizer_path=None)

        # Then
        assert_that(checkpointer.save.call_args.kwargs["optimizer"]).is_none()
        assert_that(checkpointer.save.call_args.kwargs["opt_param_scheduler"]).is_none()

    def test_optimizer_is_included_when_optimizer_path_given(self, worker, checkpointer):
        # Given
        install_into_worker(worker, mode=MODE_AUGMENT)

        # When
        worker.save_checkpoint("/durable/step_0", optimizer_path="/durable/step_0/optim")

        # Then
        assert_that(checkpointer.save.call_args.kwargs["optimizer"]).is_same_as(worker.optimizer)
        assert_that(checkpointer.save.call_args.kwargs["opt_param_scheduler"]).is_same_as(worker.scheduler)

    def test_ml_flashpoint_failure_does_not_break_the_durable_save(self, worker, checkpointer):
        # Given
        checkpointer.save.side_effect = RuntimeError("buffer pool exhausted")
        install_into_worker(worker, mode=MODE_AUGMENT)

        # When
        result = worker.save_checkpoint("/durable/step_0")

        # Then
        assert_that(result).is_equal_to("durable-result")
        assert_that(worker.durable_saves).is_length(1)


class TestReplaceMode:
    def test_keeps_every_nth_durable_save(self, worker, checkpointer):
        # Given
        install_into_worker(worker, mode=MODE_REPLACE, durable_every_n_saves=3)

        # When
        for step in range(6):
            worker.save_checkpoint(f"/durable/step_{step}")

        # Then
        assert_that(checkpointer.save.call_count).is_equal_to(6)
        assert_that([path for path, _ in worker.durable_saves]).is_equal_to(["/durable/step_2", "/durable/step_5"])

    def test_skipped_save_returns_none(self, worker, checkpointer):
        # Given
        install_into_worker(worker, mode=MODE_REPLACE, durable_every_n_saves=2)

        # When
        first = worker.save_checkpoint("/durable/step_0")
        second = worker.save_checkpoint("/durable/step_1")

        # Then
        assert_that(first).is_none()
        assert_that(second).is_equal_to("durable-result")

    def test_interval_of_one_matches_augment(self, worker, checkpointer):
        # Given
        install_into_worker(worker, mode=MODE_REPLACE, durable_every_n_saves=1)

        # When
        worker.save_checkpoint("/durable/step_0")
        worker.save_checkpoint("/durable/step_1")

        # Then
        assert_that(worker.durable_saves).is_length(2)


class TestUninstall:
    def test_restores_the_original_method(self, worker, checkpointer):
        # Given
        install_into_worker(worker, mode=MODE_REPLACE, durable_every_n_saves=100)

        # When
        uninstall_from_worker(worker)
        worker.save_checkpoint("/durable/step_0")

        # Then
        assert_that(worker.durable_saves).is_length(1)
        assert_that(get_checkpointer(worker)).is_none()

    def test_shuts_the_checkpointer_down(self, worker, checkpointer):
        # Given
        install_into_worker(worker)

        # When
        uninstall_from_worker(worker)

        # Then
        checkpointer.shutdown.assert_called_once()

    def test_uninstalling_a_clean_worker_is_a_noop(self, worker, checkpointer):
        # Given/When
        uninstall_from_worker(worker)

        # Then
        assert_that(get_checkpointer(worker)).is_none()


class TestInstallFromEnv:
    def test_disabled_by_default(self, worker, checkpointer, monkeypatch):
        # Given
        monkeypatch.delenv("MLFLASHPOINT_NEMO_RL_ENABLED", raising=False)

        # When
        result = install_from_env(worker)

        # Then
        assert_that(result).is_none()
        assert_that(get_checkpointer(worker)).is_none()

    def test_installs_when_enabled(self, worker, checkpointer, monkeypatch):
        # Given
        monkeypatch.setenv("MLFLASHPOINT_NEMO_RL_ENABLED", "true")

        # When
        result = install_from_env(worker)

        # Then
        assert_that(result).is_same_as(checkpointer)

    def test_reads_mode_and_interval(self, worker, checkpointer, monkeypatch, mocker):
        # Given
        monkeypatch.setenv("MLFLASHPOINT_NEMO_RL_ENABLED", "true")
        monkeypatch.setenv("MLFLASHPOINT_NEMO_RL_MODE", MODE_REPLACE)
        monkeypatch.setenv("MLFLASHPOINT_NEMO_RL_DURABLE_EVERY_N_SAVES", "4")
        install = mocker.spy(integration_module, "install_into_worker")

        # When
        install_from_env(worker)

        # Then
        assert_that(install.call_args.kwargs["mode"]).is_equal_to(MODE_REPLACE)
        assert_that(install.call_args.kwargs["durable_every_n_saves"]).is_equal_to(4)

    def test_respects_the_global_disable_switch(self, worker, checkpointer, monkeypatch):
        # Given
        monkeypatch.setenv("MLFLASHPOINT_NEMO_RL_ENABLED", "true")
        monkeypatch.setenv("MLFLASHPOINT_BRIDGE_ENABLED", "false")

        # When
        result = install_from_env(worker)

        # Then
        assert_that(result).is_none()
