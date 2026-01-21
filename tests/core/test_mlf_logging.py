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

import io
import logging
import re

import pytest
import torch

from ml_flashpoint.core.mlf_logging import _TRAINING_STEP, TrainingContextFormatter, get_logger, update_training_step


@pytest.fixture
def training_step_fixture():
    """Fixture to manage the training step value for tests."""
    initial_value = _TRAINING_STEP.value
    yield
    _TRAINING_STEP.value = initial_value


class TestRankFormatter:
    @pytest.fixture
    def formatter(self):
        return TrainingContextFormatter(
            "[%(asctime)s] [%(levelname)s] [Step=%(curr_step)s] [Rank %(rank)s] [%(name)s:%(lineno)d] %(message)s"
        )

    @pytest.fixture
    def record(self):
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test_module.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        return record

    @pytest.mark.parametrize("rank_value", [0, 1, 5, 8])
    def test_format_with_initialized_distributed(self, mocker, formatter, record, rank_value):
        mocker.patch("torch.distributed.is_initialized", return_value=True)
        mocker.patch("torch.distributed.get_rank", return_value=rank_value)

        formatted_message = formatter.format(record)

        assert f"[Rank {rank_value}]" in formatted_message
        torch.distributed.is_initialized.assert_called_once()
        torch.distributed.get_rank.assert_called_once()

    def test_format_with_uninitialized_distributed(self, mocker, formatter, record):
        mocker.patch("torch.distributed.is_initialized", return_value=False)

        formatted_message = formatter.format(record)

        assert "[Rank -1]" in formatted_message
        torch.distributed.is_initialized.assert_called_once()

    @pytest.mark.parametrize("step_value", [0, 1, 100])
    def test_format_with_valid_training_step(self, mocker, formatter, record, step_value, training_step_fixture):
        # Given
        mocker.patch("torch.distributed.is_initialized", return_value=False)
        with _TRAINING_STEP.get_lock():
            _TRAINING_STEP.value = step_value

        # When
        formatted_message = formatter.format(record)

        # Then
        assert f"[Step={step_value}]" in formatted_message

    @pytest.mark.parametrize("invalid_step_value", [-5, -999, -1])
    def test_format_with_invalid_training_step(
        self, mocker, formatter, record, invalid_step_value, training_step_fixture
    ):
        # Given
        mocker.patch("torch.distributed.is_initialized", return_value=False)
        with _TRAINING_STEP.get_lock():
            _TRAINING_STEP.value = invalid_step_value

        # When
        formatted_message = formatter.format(record)

        # Then
        assert f"[Step={invalid_step_value}]" in formatted_message


class TestGetLogger:
    @pytest.fixture(autouse=True)
    def cleanup_loggers(self):
        yield
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            if logger_name.startswith("test_logger"):
                logger = logging.getLogger(logger_name)
                # Remove all handlers from the logger
                for handler in list(logger.handlers):
                    logger.removeHandler(handler)
                # Then delete the logger from the manager's dictionary
                del logging.Logger.manager.loggerDict[logger_name]

    def test_get_logger_configures_handler_and_formatter(self, mocker, training_step_fixture):
        mock_stream = io.StringIO()
        mocker.patch("torch.distributed.is_initialized", return_value=True)
        mocker.patch("torch.distributed.get_rank", return_value=1)

        # Given
        update_training_step(123)

        # When
        logger = get_logger("test_logger_config_unique", stream=mock_stream)
        assert isinstance(logger, logging.Logger)
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert isinstance(handler.formatter, TrainingContextFormatter)
        assert not logger.propagate

        logger.info("Test message from logger")
        log_output = mock_stream.getvalue()
        match = re.search(
            r"\[MLF \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\ INFO "
            r"Step=123 Rank=1 test_logger_config_unique:\d+\] Test message from logger",
            log_output,
        )
        assert match is not None, f"got {log_output}"

    @pytest.mark.parametrize(
        "step_value, expected_step_str, rank_value, expected_rank_str",
        [
            (456, "456", 0, "0"),
            (None, "-1", 1, "1"),
            (-1, "-1", None, "-1"),
        ],
    )
    def test_get_logger_returns_existing_logger(
        self,
        mocker,
        training_step_fixture,
        step_value,
        expected_step_str,
        rank_value,
        expected_rank_str,
    ):
        mock_stream = io.StringIO()

        # Given
        if rank_value is not None:
            mocker.patch("torch.distributed.is_initialized", return_value=True)
            mocker.patch("torch.distributed.get_rank", return_value=rank_value)

        if step_value is not None:
            with _TRAINING_STEP.get_lock():
                _TRAINING_STEP.value = step_value

        # First call to configure
        logger1 = get_logger("test_logger_existing", stream=mock_stream)
        handler1 = logger1.handlers[0]

        # Second call should return the same logger instance and not add new handlers
        logger2 = get_logger("test_logger_existing", stream=mock_stream)
        assert logger1 is logger2
        assert len(logger2.handlers) == 1
        assert logger2.handlers[0] is handler1

        # Ensure no duplicate log output
        logger1.info("Another test message")
        log_output = mock_stream.getvalue()
        match = re.search(
            r"\[MLF \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}\ INFO "
            rf"Step={expected_step_str} Rank={expected_rank_str} test_logger_existing:\d+\] Another test message",
            log_output,
        )
        assert match is not None, f"got {log_output}"
        assert log_output.count("Another test message") == 1


class TestUpdateTrainingStep:
    def test_update_training_step(self, training_step_fixture):
        # Given
        initial_value = _TRAINING_STEP.value
        new_value = initial_value + 10

        # When
        update_training_step(new_value)

        # Then
        assert _TRAINING_STEP.value == new_value
