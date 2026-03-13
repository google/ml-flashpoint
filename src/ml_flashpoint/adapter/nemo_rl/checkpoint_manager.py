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

import os
from typing import Any, Mapping, Optional

from nemo_rl.utils.checkpoint import CheckpointManager
from typing_extensions import override

from ml_flashpoint.adapter.megatron.save_strategies import MLFlashpointMegatronAsyncSaveStrategy
from ml_flashpoint.adapter.megatron.save_utils import save_local_aware_megatron_checkpoint
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId
from ml_flashpoint.core.checkpoint_loader import MLFlashpointCheckpointLoader


class MLFlashpointRLCheckpointManager(CheckpointManager):
    """A dual checkpoint manager for NeMo/RL that coordinates ML Flashpoint saves.

    This manager overrides `init_tmp_checkpoint` to differentiate between
    a standard save (infrequent, to long-term storage) and an ML Flashpoint save
    (frequent, to tmpfs).


    Important:
      You must configure NeMo/RL's `checkpointing.save_period` to the frequency
      at which you want ML Flashpoint saves to occur. This ensures the algorithm
      loop triggers `init_tmp_checkpoint` frequently enough.
      Then, pass your desired standard save period to this manager via `standard_save_period`.
    """

    def __init__(
        self,
        base_checkpointer: CheckpointManager,
        policy: Any,
        flashpoint_base_container: str,
        standard_save_period: int,
        save_strategy: MLFlashpointMegatronAsyncSaveStrategy,
        checkpoint_loader: MLFlashpointCheckpointLoader,
    ):
        """Initializes the MLFlashpointRLCheckpointManager.

        Args:
            base_checkpointer: The original NeMo/RL CheckpointManager.
            policy: The NeMo/RL policy worker (e.g., MegatronPolicyWorker).
            flashpoint_base_container: The base container ID / path for MLF checkpoints.
            standard_save_period: How often to take standard saves (measured in steps).
            save_strategy: The MLFlashpointMegatronAsyncSaveStrategy for asynchronous background saves.
            checkpoint_loader: The MLFlashpointCheckpointLoader for resolving latest MLF saves.
        """
        self._base_checkpointer = base_checkpointer
        self.policy = policy
        self.flashpoint_base_container = CheckpointContainerId(flashpoint_base_container)
        self.standard_save_period = standard_save_period
        self.save_strategy = save_strategy
        self.checkpoint_loader = checkpoint_loader

        # Track the active save mode ("std" or "mlf")
        self._current_save_mode: Optional[str] = None

        # Monkey-patch the policy's save_checkpoint method
        self._original_save_checkpoint = policy.save_checkpoint

        # Need to bind the interception method to the manager context while keeping policy self
        def _intercepted_save_checkpoint_wrapper(*args, **kwargs):
            return self._intercepted_save_checkpoint(*args, **kwargs)

        policy.save_checkpoint = _intercepted_save_checkpoint_wrapper

    def __getattr__(self, name: str) -> Any:
        """Dynamically delegate missing attributes to the base checkpointer.

        This allows the algorithm loop to transparently access properties like
        `checkpoint_dir`, `keep_top_k`, `metric_name`, etc., without us needing
        to manually map them or call super().__init__() with the full config.
        """
        # Prevent infinite recursion if base_checkpointer hasn't been set yet
        if name == "base_checkpointer":
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        return getattr(self._base_checkpointer, name)

    @override
    def init_tmp_checkpoint(
        self,
        step: int,
        training_info: Mapping[str, Any],
        run_config: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Initializes the checkpoint directory based on the save frequency."""
        # Check standard save
        if step % self.standard_save_period == 0:
            self._current_save_mode = "std"
            # Return string since PathLike is expected/supported
            return str(self._base_checkpointer.init_tmp_checkpoint(step, training_info, run_config))

        # Otherwise, assume it's an ML Flashpoint save because the loop triggered it
        self._current_save_mode = "mlf"

        # We need a proper MLF CheckpointContainerId child path
        mlf_path = CheckpointContainerId.create_child(
            self.flashpoint_base_container, CheckpointContainerId.format_version_container(step)
        )

        # Create tmpfs dirs to allow standard file saving to succeed:
        os.makedirs(str(mlf_path), exist_ok=True)
        return str(mlf_path)

    @override
    def finalize_checkpoint(self, checkpoint_path: str) -> None:
        """Finalizes the checkpoint depending on the save mode."""
        if self._current_save_mode == "std":
            self._base_checkpointer.finalize_checkpoint(checkpoint_path)
        else:
            # We don't rename MLF checkpoints, ML Flashpoint manages their lifecycle mapping natively.
            pass

    @override
    def get_best_checkpoint_path(self) -> Optional[str]:
        return self._base_checkpointer.get_best_checkpoint_path()

    @override
    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Returns the path to the freshest available checkpoint (Standard or MLF)."""
        base_path = self._base_checkpointer.get_latest_checkpoint_path()

        # Check for ML Flashpoint checkpoints
        latest_mlf_container = self.checkpoint_loader.get_latest_complete_checkpoint(self.flashpoint_base_container)

        if latest_mlf_container is None:
            return base_path

        mlf_path = latest_mlf_container.data
        mlf_step = CheckpointContainerId.parse_version_container_step(os.path.basename(mlf_path))

        # CheckpointContainerId.parse_version_container_step returns None if it fails to parse
        if mlf_step is None:
            return base_path

        if base_path is None:
            return mlf_path

        # We have both. Compare step numbers.
        # base_path typically looks like "/path/to/checkpoints/step_100" or similar.
        # If the RL algorithm manages to get a step out of it, we want ours to be >.
        # Instead of strict parsing which depends on NeMo RL formatting, we can check
        # load_training_info which returns a dict like {"step": 100} in NeMo RL.
        try:
            base_info = self.load_training_info(base_path)
            base_step = base_info.get("step", -1) if base_info else -1
        except Exception:
            base_step = -1

        if mlf_step > base_step:
            return mlf_path

        return base_path

    @override
    def load_training_info(self, checkpoint_path: Optional[str] = None) -> Optional[dict]:
        return self._base_checkpointer.load_training_info(checkpoint_path)

    @override
    def remove_old_checkpoints(self, exclude_latest: bool = True) -> None:
        return self._base_checkpointer.remove_old_checkpoints(exclude_latest)

    def _intercepted_save_checkpoint(
        self,
        *args,
        **kwargs,
    ):
        """Replaces the policy's native save_checkpoint logic during an MLF save."""
        if self._current_save_mode == "std":
            return self._original_save_checkpoint(*args, **kwargs)

        # Extract paths
        weights_path = kwargs.get("weights_path")
        if weights_path is None and len(args) > 0:
            weights_path = args[0]

        optimizer_path = kwargs.get("optimizer_path")
        if optimizer_path is None and len(args) > 1:
            optimizer_path = args[1]

        if not weights_path:
            raise ValueError("weights_path must be provided to save_checkpoint.")

        # Ensure model is in eval mode for consistent saving, mimicking original logic
        is_training = self.policy.model.training
        if not is_training:
            self.policy.model.eval()

        has_hook = hasattr(self.policy, "should_disable_forward_pre_hook")
        if has_hook and getattr(self.policy, "should_disable_forward_pre_hook"):
            self.policy.disable_forward_pre_hook()

        # Build checkpoint dict matching Megatron Core dist_checkpointing save format
        checkpoint_dict = {
            "model": [self.policy.model],
            "state": self.policy.mcore_state,
        }

        if optimizer_path is not None:
            if hasattr(self.policy, "optimizer") and self.policy.optimizer is not None:
                checkpoint_dict["optimizer"] = self.policy.optimizer
            if hasattr(self.policy, "scheduler") and self.policy.scheduler is not None:
                checkpoint_dict["opt_param_scheduler"] = self.policy.scheduler

        # Optional metadata that Megatron may capture
        if hasattr(self.policy.mcore_state, "train_state"):
            checkpoint_dict["num_floating_point_operations_so_far"] = (
                self.policy.mcore_state.train_state.floating_point_operations_so_far
            )
        if hasattr(self.policy, "checkpointing_context"):
            checkpoint_dict["checkpointing_context"] = self.policy.checkpointing_context

        save_local_aware_megatron_checkpoint(
            checkpoint=checkpoint_dict, checkpoint_dir=weights_path, save_strategy=self.save_strategy, async_save=True
        )

        has_hook = hasattr(self.policy, "should_disable_forward_pre_hook")
        if has_hook and getattr(self.policy, "should_disable_forward_pre_hook"):
            self.policy.enable_forward_pre_hook()

        if not is_training:
            self.policy.model.train()
