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

from ml_flashpoint.adapter.nemo_rl.wrapper_util import wrap_rl_components_with_mlflashpoint


def test_wrap_rl_components_with_mlflashpoint(mocker):
    """Test that it correctly instantiates MLFlashpointRLCheckpointManager."""
    # Given
    mock_manager_cls = mocker.patch("ml_flashpoint.adapter.nemo_rl.wrapper_util.MLFlashpointRLCheckpointManager")
    checkpointer = mocker.MagicMock()
    flashpoint_base_container = "/tmp/mlf"
    standard_save_period = 1000
    save_strategy = mocker.MagicMock()
    checkpoint_loader = mocker.MagicMock()

    # When
    actual_wrapped = wrap_rl_components_with_mlflashpoint(
        checkpointer=checkpointer,
        flashpoint_base_container=flashpoint_base_container,
        standard_save_period=standard_save_period,
        save_strategy=save_strategy,
        checkpoint_loader=checkpoint_loader,
    )

    # Then
    mock_manager_cls.assert_called_once_with(
        base_checkpointer=checkpointer,
        flashpoint_base_container=flashpoint_base_container,
        standard_save_period=standard_save_period,
        save_strategy=save_strategy,
        checkpoint_loader=checkpoint_loader,
    )
    assert actual_wrapped == mock_manager_cls.return_value
