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

from ml_flashpoint.adapter.megatron_bridge import config as config_module
from ml_flashpoint.adapter.megatron_bridge.config import (
    DEFAULT_BASE_CONTAINER,
    MLFlashpointBridgeConfig,
    configure,
    get_config,
    reset_configuration,
)
from ml_flashpoint.core.checkpoint_saver import DEFAULT_INITIAL_BUFFER_SIZE_BYTES


@pytest.fixture(autouse=True)
def _clean_registration():
    reset_configuration()
    yield
    reset_configuration()


class TestMLFlashpointBridgeConfig:
    def test_defaults(self):
        # Given/When
        config = MLFlashpointBridgeConfig()

        # Then
        assert_that(config.enabled).is_true()
        assert_that(config.base_container).is_equal_to(DEFAULT_BASE_CONTAINER)
        assert_that(config.async_save).is_true()
        assert_that(config.write_thread_count).is_equal_to(1)
        assert_that(config.initial_write_buffer_size_bytes).is_equal_to(DEFAULT_INITIAL_BUFFER_SIZE_BYTES)
        assert_that(config.use_optimized_save).is_true()
        assert_that(config.use_cached_ckpt_structure).is_false()
        assert_that(config.use_fully_parallel_wrapper).is_true()
        assert_that(config.keep_checkpoints_on_finalize).is_false()

    def test_is_frozen(self):
        # Given
        config = MLFlashpointBridgeConfig()

        # When/Then
        with pytest.raises(Exception):
            config.enabled = False

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"base_container": ""},
            {"write_thread_count": 0},
            {"write_thread_count": -3},
            {"initial_write_buffer_size_bytes": 0},
            {"initial_write_buffer_size_bytes": -1},
        ],
    )
    def test_rejects_invalid_values(self, kwargs):
        # Given/When/Then
        with pytest.raises(ValueError):
            MLFlashpointBridgeConfig(**kwargs)


class TestFromEnv:
    def test_uses_defaults_when_env_is_empty(self, monkeypatch):
        # Given
        for name in list(os_environ_keys()):
            monkeypatch.delenv(name, raising=False)

        # When
        config = MLFlashpointBridgeConfig.from_env()

        # Then
        assert_that(config).is_equal_to(MLFlashpointBridgeConfig())

    def test_reads_every_field(self, monkeypatch):
        # Given
        monkeypatch.setenv("MLFLASHPOINT_BRIDGE_ENABLED", "false")
        monkeypatch.setenv("MLFLASHPOINT_BASE_CONTAINER", "/mnt/local/mlf")
        monkeypatch.setenv("MLFLASHPOINT_ASYNC_SAVE", "false")
        monkeypatch.setenv("MLFLASHPOINT_WRITE_THREAD_COUNT", "4")
        monkeypatch.setenv("MLFLASHPOINT_INITIAL_WRITE_BUFFER_SIZE_BYTES", "2048")
        monkeypatch.setenv("MLFLASHPOINT_USE_OPTIMIZED_SAVE", "false")
        monkeypatch.setenv("MLFLASHPOINT_USE_CACHED_CKPT_STRUCTURE", "true")
        monkeypatch.setenv("MLFLASHPOINT_USE_FULLY_PARALLEL_WRAPPER", "false")
        monkeypatch.setenv("MLFLASHPOINT_KEEP_CHECKPOINTS_ON_FINALIZE", "true")

        # When
        config = MLFlashpointBridgeConfig.from_env()

        # Then
        assert_that(config.enabled).is_false()
        assert_that(config.base_container).is_equal_to("/mnt/local/mlf")
        assert_that(config.async_save).is_false()
        assert_that(config.write_thread_count).is_equal_to(4)
        assert_that(config.initial_write_buffer_size_bytes).is_equal_to(2048)
        assert_that(config.use_optimized_save).is_false()
        assert_that(config.use_cached_ckpt_structure).is_true()
        assert_that(config.use_fully_parallel_wrapper).is_false()
        assert_that(config.keep_checkpoints_on_finalize).is_true()

    def test_non_integer_value_falls_back_to_default(self, monkeypatch):
        # Given
        monkeypatch.setenv("MLFLASHPOINT_WRITE_THREAD_COUNT", "not-a-number")

        # When
        config = MLFlashpointBridgeConfig.from_env()

        # Then
        assert_that(config.write_thread_count).is_equal_to(1)

    def test_invalid_env_value_still_validated(self, monkeypatch):
        # Given
        monkeypatch.setenv("MLFLASHPOINT_WRITE_THREAD_COUNT", "0")

        # When/Then
        with pytest.raises(ValueError):
            MLFlashpointBridgeConfig.from_env()


class TestConfigure:
    def test_get_config_reads_env_when_unregistered(self, monkeypatch):
        # Given
        monkeypatch.setenv("MLFLASHPOINT_BASE_CONTAINER", "/from/env")

        # When
        config = get_config()

        # Then
        assert_that(config.base_container).is_equal_to("/from/env")

    def test_registered_config_wins_over_env(self, monkeypatch):
        # Given
        monkeypatch.setenv("MLFLASHPOINT_BASE_CONTAINER", "/from/env")
        configure(MLFlashpointBridgeConfig(base_container="/registered"))

        # When
        config = get_config()

        # Then
        assert_that(config.base_container).is_equal_to("/registered")

    def test_reset_restores_env_lookup(self, monkeypatch):
        # Given
        monkeypatch.setenv("MLFLASHPOINT_BASE_CONTAINER", "/from/env")
        configure(MLFlashpointBridgeConfig(base_container="/registered"))

        # When
        reset_configuration()

        # Then
        assert_that(get_config().base_container).is_equal_to("/from/env")

    def test_module_state_starts_clean(self):
        # Given/When/Then
        assert_that(config_module._CONFIGURED).is_none()


def os_environ_keys():
    """Returns every ``MLFLASHPOINT_`` variable currently set."""
    import os

    return [name for name in os.environ if name.startswith("MLFLASHPOINT_")]
