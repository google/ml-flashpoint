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

"""Stubs Megatron Bridge so the adapter can be unit tested without it.

The Megatron Bridge adapter binds Bridge symbols at import time. Bridge itself
pulls in the full NVIDIA training stack (Transformer Engine, ModelOpt, NVRx),
which is neither installable nor runnable in a CPU-only test environment, so
this installs a minimal fake package tree when the real one is absent.

Individual tests still patch these symbols where they are bound in the adapter
modules, so the fakes only have to satisfy ``import``.
"""

import importlib.util
import sys
import types
from typing import Any


def _bridge_is_installed() -> bool:
    """Whether the real Megatron Bridge can be imported.

    Returns:
        True when ``megatron.bridge.training.checkpointing`` is importable.
    """
    try:
        return importlib.util.find_spec("megatron.bridge.training.checkpointing") is not None
    except (ImportError, ValueError):
        return False


class _FakeTrainState:
    """Minimal stand-in for Megatron Bridge's ``TrainState``."""

    def __init__(self, step: int = 0):
        self.step = step
        self.consumed_train_samples = 0
        self.skipped_train_samples = 0
        self.consumed_valid_samples = 0
        self.floating_point_operations_so_far = 0

    def state_dict(self) -> dict[str, Any]:
        """Returns the serializable counters."""
        return {
            "step": self.step,
            "consumed_train_samples": self.consumed_train_samples,
            "skipped_train_samples": self.skipped_train_samples,
            "consumed_valid_samples": self.consumed_valid_samples,
            "floating_point_operations_so_far": self.floating_point_operations_so_far,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restores counters from ``state_dict``."""
        for key, value in state_dict.items():
            setattr(self, key, value)


def _module(name: str) -> types.ModuleType:
    """Creates and registers a module.

    Args:
        name: Fully qualified module name.

    Returns:
        The registered module.
    """
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _install_fake_bridge() -> None:
    """Registers a fake ``megatron.bridge`` package tree in ``sys.modules``."""
    # `megatron` is a namespace package shared with the real megatron.core, so
    # only reuse it -- never replace it.
    megatron = sys.modules.get("megatron")
    if megatron is None:
        megatron = _module("megatron")
        megatron.__path__ = []

    bridge = _module("megatron.bridge")
    bridge.__path__ = []
    megatron.bridge = bridge

    training = _module("megatron.bridge.training")
    training.__path__ = []
    bridge.training = training

    checkpointing = _module("megatron.bridge.training.checkpointing")
    checkpointing.init_checkpointing_context = lambda checkpoint_config: {}
    checkpointing.load_checkpoint = lambda **kwargs: (0, 0)
    checkpointing.maybe_finalize_async_save = lambda **kwargs: None
    checkpointing.save_checkpoint = lambda **kwargs: None
    checkpointing.generate_state_dict = lambda *args, **kwargs: {}
    checkpointing.get_rng_state = lambda *args, **kwargs: None
    checkpointing.set_checkpoint_version = lambda value: None
    checkpointing._build_sharded_state_dict_metadata = lambda use_distributed_optimizer, cfg: {}
    checkpointing._load_model_state_dict = lambda module, state_dict, strict: None
    training.checkpointing = checkpointing

    state = _module("megatron.bridge.training.state")
    state.TrainState = _FakeTrainState
    training.state = state

    training_utils = _module("megatron.bridge.training.utils")
    training_utils.__path__ = []
    training.utils = training_utils

    pg_utils = _module("megatron.bridge.training.utils.pg_utils")
    pg_utils.get_pg_collection = lambda model: None
    training_utils.pg_utils = pg_utils

    bridge_utils = _module("megatron.bridge.utils")
    bridge_utils.__path__ = []
    bridge.utils = bridge_utils

    instantiate_utils = _module("megatron.bridge.utils.instantiate_utils")
    instantiate_utils.registered_prefixes = []
    instantiate_utils.register_allowed_target_prefix = instantiate_utils.registered_prefixes.append
    bridge_utils.instantiate_utils = instantiate_utils


MEGATRON_BRIDGE_IS_FAKE = not _bridge_is_installed()
"""Whether these tests run against the fake Bridge rather than the real one."""

if MEGATRON_BRIDGE_IS_FAKE:
    _install_fake_bridge()
