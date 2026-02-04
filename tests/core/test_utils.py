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

import logging
import re
import time

import pytest
import torch

from ml_flashpoint.core import utils


class TestLogExecutionTime:
    def test_logs_time_on_successful_execution(self, mocker):
        # Given
        mock_logger = mocker.MagicMock(spec=logging.Logger)
        op_name = "test_op"
        mocker.patch("time.perf_counter", side_effect=[0, 0.1234])

        # When
        with utils.log_execution_time(mock_logger, op_name):
            pass

        # Then
        mock_logger.log.assert_called_once_with(logging.DEBUG, "%s took %.4fs", op_name, 0.1234)

    def test_timing_accuracy_with_delay(self, mocker):
        # Given
        mock_logger = mocker.MagicMock(spec=logging.Logger)
        op_name = "delayed_op"
        delay = 0.1
        mocker.patch("time.perf_counter", side_effect=[0, delay])

        # When
        with utils.log_execution_time(mock_logger, op_name):
            time.sleep(delay)

        # Then
        mock_logger.log.assert_called_once_with(logging.DEBUG, "%s took %.4fs", op_name, delay)

    def test_logs_time_when_exception_is_raised(self, mocker):
        # Given
        mock_logger = mocker.MagicMock(spec=logging.Logger)
        op_name = "failing_op"
        error_message = "test error"
        mocker.patch("time.perf_counter", side_effect=[0, 0.1234])

        # When/Then
        with pytest.raises(ValueError, match=error_message):
            with utils.log_execution_time(mock_logger, op_name):
                raise ValueError(error_message)

        mock_logger.log.assert_called_once_with(logging.DEBUG, "%s took %.4fs", op_name, 0.1234)

    @pytest.mark.parametrize("log_level", [logging.INFO, logging.WARNING, logging.DEBUG])
    def test_logs_with_different_levels(self, mocker, log_level):
        # Given
        mock_logger = mocker.MagicMock(spec=logging.Logger)
        op_name = "test_op"
        mocker.patch("time.perf_counter", side_effect=[0, 1])

        # When
        with utils.log_execution_time(mock_logger, op_name, level=log_level):
            pass

        # Then
        mock_logger.log.assert_called_once_with(log_level, "%s took %.4fs", op_name, 1)


@pytest.mark.parametrize(
    "env_vars, expected",
    [
        ({"NNODES": "4"}, 4),
        ({"SLURM_NNODES": "3"}, 3),
        ({"NNODES": "4", "SLURM_NNODES": "3"}, 4),
    ],
)
def test_get_num_of_nodes_torch_only_with_env_vars(env_vars, expected, monkeypatch):
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)
    assert utils.get_num_of_nodes() == expected


def test_get_num_of_nodes_torch_only_no_dist(monkeypatch):
    monkeypatch.delenv("NNODES", raising=False)
    monkeypatch.delenv("SLURM_NNODES", raising=False)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    assert utils.get_num_of_nodes() == 1


def test_get_num_of_nodes_torch_only_dist_initialized(monkeypatch):
    monkeypatch.delenv("NNODES", raising=False)
    monkeypatch.delenv("SLURM_NNODES", raising=False)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 16)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    assert utils.get_num_of_nodes() == 2


def test_get_num_of_nodes_torch_only_cpu_only_training_raises_error(monkeypatch):
    monkeypatch.delenv("NNODES", raising=False)
    monkeypatch.delenv("SLURM_NNODES", raising=False)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Cannot determine number of accelerators for CPU-only training"):
        utils.get_num_of_nodes()


def test_get_num_of_nodes_torch_only_no_cuda_devices_raises_error(monkeypatch):
    monkeypatch.delenv("NNODES", raising=False)
    monkeypatch.delenv("SLURM_NNODES", raising=False)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match=re.escape("torch.cuda.device_count() returned 0.")):
        utils.get_num_of_nodes()


def test_get_num_of_nodes_torch_only_inconsistent_setup_raises_error(monkeypatch):
    monkeypatch.delenv("NNODES", raising=False)
    monkeypatch.delenv("SLURM_NNODES", raising=False)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 8)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    with pytest.raises(RuntimeError, match="Inconsistent setup"):
        utils.get_num_of_nodes()


@pytest.mark.parametrize(
    "env_vars, var_name, default_val, expected",
    [
        ({"MLFLASHPOINT_TEST_VAR": "True"}, "TEST_VAR", False, True),
        ({"MLFLASHPOINT_TEST_VAR": "true"}, "TEST_VAR", False, True),
        ({"MLFLASHPOINT_TEST_VAR": "False"}, "TEST_VAR", True, False),
        ({"MLFLASHPOINT_TEST_VAR": "false"}, "TEST_VAR", True, False),
        ({}, "TEST_VAR", True, True),
        ({}, "TEST_VAR", False, False),
        ({"MLFLASHPOINT_TEST_VAR": "123"}, "TEST_VAR", False, False),
        ({"OTHERPREFIX_TEST_VAR": "True"}, "TEST_VAR", False, False),
        ({"OTHERPREFIX_TEST_VAR": "False"}, "TEST_VAR", True, True),
    ],
)
def test_get_env_val_bool(env_vars, var_name, default_val, expected, monkeypatch):
    # Given
    for k, v in env_vars.items():
        monkeypatch.setenv(k, v)

    # When
    result = utils.get_env_val_bool(var_name, default_val)

    # Then
    assert result == expected


class TestGetEnvValStr:
    @pytest.mark.parametrize(
        "env_vars, var_name, default_val, expected",
        [
            ({"MLFLASHPOINT_TEST_VAR": "hello"}, "TEST_VAR", "default", "hello"),
            ({"MLFLASHPOINT_TEST_VAR": "hello_world"}, "TEST_VAR", "default", "hello_world"),
            ({"MLFLASHPOINT_TEST_VAR": "12345"}, "TEST_VAR", "0", "12345"),
            ({"MLFLASHPOINT_TEST_VAR": ""}, "TEST_VAR", "default", ""),
            ({}, "TEST_VAR", "default", "default"),
            ({"OTHERPREFIX_TEST_VAR": "world"}, "TEST_VAR", "default", "default"),
            ({"TEST_VAR": "no_prefix"}, "TEST_VAR", "default", "default"),
        ],
    )
    def test_get_env_val_str(self, env_vars, var_name, default_val, expected, monkeypatch):
        # Given
        for k, v in env_vars.items():
            monkeypatch.setenv(k, v)

        # When
        result = utils.get_env_val_str(var_name, default_val)

        # Then
        assert result == expected


class TestGetEnvValInt:
    @pytest.mark.parametrize(
        "env_vars, var_name, default_val, expected",
        [
            ({"MLFLASHPOINT_TEST_VAR": "123"}, "TEST_VAR", 0, 123),
            ({"MLFLASHPOINT_TEST_VAR": "-45"}, "TEST_VAR", 10, -45),
            ({"MLFLASHPOINT_TEST_VAR": "0"}, "TEST_VAR", 5, 0),
            ({}, "TEST_VAR", 99, 99),
            ({"MLFLASHPOINT_TEST_VAR": "abc"}, "TEST_VAR", 100, 100),
            ({"MLFLASHPOINT_TEST_VAR": "12.3"}, "TEST_VAR", 200, 200),
            ({"MLFLASHPOINT_TEST_VAR": ""}, "TEST_VAR", 300, 300),
            ({"OTHERPREFIX_TEST_VAR": "456"}, "TEST_VAR", 0, 0),
            ({"TEST_VAR": "789"}, "TEST_VAR", 0, 0),
        ],
    )
    def test_get_env_val_int(self, env_vars, var_name, default_val, expected, monkeypatch):
        # Given
        for k, v in env_vars.items():
            monkeypatch.setenv(k, v)

        # When
        result = utils.get_env_val_int(var_name, default_val)

        # Then
        assert result == expected


class TestGetAcceleratorCount:
    def test_get_accelerator_count_cuda_available(self, mocker):
        # Given
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.device_count", return_value=4)

        # When
        count = utils.get_accelerator_count()

        # Then
        assert count == 4

    def test_get_accelerator_count_cuda_unavailable(self, mocker):
        # Given
        mocker.patch("torch.cuda.is_available", return_value=False)

        # When/Then
        with pytest.raises(RuntimeError, match="Cannot determine number of accelerators for CPU-only training"):
            utils.get_accelerator_count()
