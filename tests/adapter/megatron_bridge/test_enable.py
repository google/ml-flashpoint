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
import sys
import types
from typing import Optional

import pytest
from assertpy import assert_that

import ml_flashpoint.adapter.megatron_bridge as adapter
from ml_flashpoint.adapter.megatron_bridge import CUSTOM_MANAGER_CLASS, MLFlashpointBridgeConfig
from ml_flashpoint.adapter.megatron_bridge.config import reset_configuration


@dataclasses.dataclass
class FakeCheckpointConfig:
    save: str = "/durable/checkpoints"
    save_interval: int = 100
    non_persistent_ckpt_type: Optional[str] = None
    non_persistent_save_interval: Optional[int] = None
    custom_manager_class: Optional[str] = None


@pytest.fixture(autouse=True)
def _clean_registration():
    reset_configuration()
    yield
    reset_configuration()


@pytest.fixture
def instantiate_utils(mocker):
    """A stand-in for Megatron Bridge's target allowlist module."""
    module = types.ModuleType("megatron.bridge.utils.instantiate_utils")
    module.prefixes = []
    module.register_allowed_target_prefix = module.prefixes.append
    mocker.patch.dict(sys.modules, {"megatron.bridge.utils.instantiate_utils": module})
    return module


class TestRegisterWithMegatronBridge:
    def test_registers_the_package_prefix(self, instantiate_utils):
        # Given/When
        adapter.register_with_megatron_bridge()

        # Then
        assert_that(instantiate_utils.prefixes).contains("ml_flashpoint")

    def test_is_idempotent(self, instantiate_utils):
        # Given/When
        adapter.register_with_megatron_bridge()
        adapter.register_with_megatron_bridge()

        # Then: registering twice is harmless; Bridge de-duplicates prefixes.
        assert_that(set(instantiate_utils.prefixes)).is_equal_to({"ml_flashpoint"})

    def test_missing_helper_is_tolerated(self, mocker):
        # Given a Bridge build without the allowlist helper.
        module = types.ModuleType("megatron.bridge.utils.instantiate_utils")
        mocker.patch.dict(sys.modules, {"megatron.bridge.utils.instantiate_utils": module})

        # When/Then: no exception, just a warning.
        adapter.register_with_megatron_bridge()


class TestEnable:
    def test_points_the_config_at_the_adapter(self, instantiate_utils):
        # Given
        config = FakeCheckpointConfig()

        # When
        adapter.enable(config, non_persistent_save_interval=20)

        # Then
        assert_that(config.custom_manager_class).is_equal_to(CUSTOM_MANAGER_CLASS)
        assert_that(config.non_persistent_ckpt_type).is_equal_to("local")
        assert_that(config.non_persistent_save_interval).is_equal_to(20)

    def test_leaves_the_durable_cadence_alone(self, instantiate_utils):
        # Given
        config = FakeCheckpointConfig(save="/durable/checkpoints", save_interval=100)

        # When
        adapter.enable(config, non_persistent_save_interval=20)

        # Then
        assert_that(config.save).is_equal_to("/durable/checkpoints")
        assert_that(config.save_interval).is_equal_to(100)

    def test_registers_the_allowlist_prefix(self, instantiate_utils):
        # Given
        config = FakeCheckpointConfig()

        # When
        adapter.enable(config, non_persistent_save_interval=5)

        # Then
        assert_that(instantiate_utils.prefixes).contains("ml_flashpoint")

    def test_registers_the_supplied_ml_flashpoint_config(self, instantiate_utils, tmp_path):
        # Given
        mlf_config = MLFlashpointBridgeConfig(base_container=str(tmp_path / "mlf"))

        # When
        adapter.enable(FakeCheckpointConfig(), non_persistent_save_interval=5, mlf_config=mlf_config)

        # Then
        assert_that(adapter.get_config()).is_equal_to(mlf_config)

    @pytest.mark.parametrize("interval", [0, -1])
    def test_rejects_a_non_positive_interval(self, instantiate_utils, interval):
        # Given/When/Then
        with pytest.raises(ValueError, match="non_persistent_save_interval"):
            adapter.enable(FakeCheckpointConfig(), non_persistent_save_interval=interval)

    def test_custom_manager_class_resolves_to_the_manager(self):
        # Given
        module_path, class_name = CUSTOM_MANAGER_CLASS.rsplit(".", 1)

        # When
        module = __import__(module_path, fromlist=[class_name])

        # Then
        assert_that(getattr(module, class_name)).is_same_as(adapter.MLFlashpointBridgeCheckpointManager)
