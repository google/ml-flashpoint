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
from typing import Any, Optional

import pytest
import torch
from assertpy import assert_that

from ml_flashpoint.adapter.megatron_bridge import bridge_state


@dataclasses.dataclass
class FakeCheckpointConfig:
    ckpt_format: str = "torch_dist"
    save_rng: bool = True
    load_rng: bool = True
    save_optim: bool = True
    load_optim: bool = True
    finetune: bool = False


@dataclasses.dataclass
class FakeTrainState:
    step: int = 0
    consumed_train_samples: int = 0
    floating_point_operations_so_far: int = 0

    def state_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for key, value in state_dict.items():
            setattr(self, key, value)


@dataclasses.dataclass
class FakeGlobalState:
    cfg: Any
    train_state: FakeTrainState = dataclasses.field(default_factory=FakeTrainState)


@dataclasses.dataclass
class FakeSaveContext:
    state: FakeGlobalState
    model: list
    optimizer: Any = None
    opt_param_scheduler: Any = None
    num_floating_point_operations_so_far: int = 0
    train_data_iterator: Any = None
    non_persistent_ckpt: bool = True
    pg_collection: Any = None
    module_name: Optional[str] = None


@dataclasses.dataclass
class FakeLoadContext:
    state: FakeGlobalState
    model: list
    optimizer: Any = None
    opt_param_scheduler: Any = None
    strict: bool = True
    skip_load_to_model_and_opt: bool = False
    pg_collection: Any = None
    module_name: Optional[str] = None


def _namespace(**kwargs):
    return type("Namespace", (), kwargs)()


@pytest.fixture
def cfg():
    return _namespace(
        checkpoint=FakeCheckpointConfig(),
        optimizer=_namespace(use_distributed_optimizer=True),
        rng=_namespace(data_parallel_random_init=False),
    )


@pytest.fixture
def pg_collection(mocker):
    collection = mocker.MagicMock()
    collection.dp.rank.return_value = 0
    return collection


@pytest.fixture
def save_ctx(cfg, pg_collection):
    return FakeSaveContext(
        state=FakeGlobalState(cfg=cfg, train_state=FakeTrainState(step=25)),
        model=[object()],
        num_floating_point_operations_so_far=987,
        pg_collection=pg_collection,
    )


@pytest.fixture
def load_ctx(cfg, pg_collection):
    return FakeLoadContext(
        state=FakeGlobalState(cfg=cfg),
        model=[object()],
        pg_collection=pg_collection,
    )


@pytest.fixture(autouse=True)
def _bridge_helpers(mocker):
    """Neutralizes the Bridge symbols the helpers call into."""
    mocker.patch.object(bridge_state, "get_rng_state", return_value="rng")
    mocker.patch.object(bridge_state, "generate_state_dict", return_value={"model": {}})
    mocker.patch.object(bridge_state, "get_rerun_state_machine", return_value=mocker.MagicMock())
    mocker.patch.object(bridge_state, "unwrap_model", side_effect=lambda model: list(model))
    mocker.patch.object(bridge_state, "build_sharded_state_dict_metadata", return_value={})


class TestUnwrapModel:
    def test_prefers_megatron_core(self, mocker):
        # Given
        mocker.stopall()
        module = types.ModuleType("megatron.core.utils")
        module.unwrap_model = lambda model: ["unwrapped"]
        mocker.patch.dict(sys.modules, {"megatron.core.utils": module})

        # When
        result = bridge_state.unwrap_model([object()])

        # Then
        assert_that(result).is_equal_to(["unwrapped"])

    def test_falls_back_to_megatron_training(self, mocker):
        # Given
        mocker.stopall()
        core_utils = types.ModuleType("megatron.core.utils")
        training_utils = types.ModuleType("megatron.training.utils")
        training_utils.unwrap_model = lambda model: ["legacy"]
        mocker.patch.dict(
            sys.modules,
            {"megatron.core.utils": core_utils, "megatron.training.utils": training_utils},
        )

        # When
        result = bridge_state.unwrap_model([object()])

        # Then
        assert_that(result).is_equal_to(["legacy"])

    def test_wraps_a_single_module_in_a_list(self, mocker):
        # Given
        mocker.stopall()
        module = types.ModuleType("megatron.core.utils")
        module.unwrap_model = lambda model: "single"
        mocker.patch.dict(sys.modules, {"megatron.core.utils": module})

        # When
        result = bridge_state.unwrap_model(object())

        # Then
        assert_that(result).is_equal_to(["single"])

    def test_raises_a_clear_error_when_unavailable(self, mocker):
        # Given
        mocker.stopall()
        core_utils = types.ModuleType("megatron.core.utils")
        mocker.patch.dict(sys.modules, {"megatron.core.utils": core_utils})
        mocker.patch.dict(sys.modules, {"megatron.training.utils": types.ModuleType("megatron.training.utils")})

        # When/Then
        with pytest.raises(RuntimeError, match="Could not resolve unwrap_model"):
            bridge_state.unwrap_model([object()])


class TestBuildSaveStateDict:
    def test_embeds_train_state_flops_and_metadata(self, save_ctx):
        # Given/When
        state_dict = bridge_state.build_save_state_dict(save_ctx, step=25, num_floating_point_operations_so_far=987)

        # Then
        assert_that(state_dict).contains_key(bridge_state.TRAIN_STATE_KEY)
        assert_that(state_dict[bridge_state.FLOPS_KEY]).is_equal_to(987)
        assert_that(state_dict).contains_key(bridge_state.CONTENT_METADATA_KEY)

    def test_passes_the_step_as_the_iteration(self, save_ctx):
        # Given/When
        bridge_state.build_save_state_dict(save_ctx, step=25, num_floating_point_operations_so_far=0)

        # Then
        assert_that(bridge_state.generate_state_dict.call_args.kwargs["iteration"]).is_equal_to(25)

    def test_skips_rng_collection_when_disabled(self, save_ctx):
        # Given
        save_ctx.state.cfg.checkpoint.save_rng = False

        # When
        bridge_state.build_save_state_dict(save_ctx, step=1, num_floating_point_operations_so_far=0)

        # Then
        bridge_state.get_rng_state.assert_not_called()

    def test_supplies_the_process_group_to_sharding_but_not_the_checkpoint(self, save_ctx, pg_collection):
        # Given/When
        state_dict = bridge_state.build_save_state_dict(save_ctx, step=1, num_floating_point_operations_so_far=0)

        # Then
        metadata = bridge_state.generate_state_dict.call_args.kwargs["optim_sd_kwargs"]["metadata"]
        assert_that(metadata["dp_cp_group"]).is_equal_to(pg_collection.dp_cp)
        assert_that(state_dict[bridge_state.CONTENT_METADATA_KEY]).does_not_contain_key("dp_cp_group")


class TestBuildLoadStateDict:
    def test_marks_the_state_dict_as_loading(self, load_ctx):
        # Given/When
        bridge_state.build_load_state_dict(load_ctx)

        # Then
        assert_that(bridge_state.generate_state_dict.call_args.kwargs["optim_sd_kwargs"]["is_loading"]).is_true()

    def test_omits_the_optimizer_when_load_optim_is_off(self, load_ctx, mocker):
        # Given
        load_ctx.optimizer = mocker.MagicMock()
        load_ctx.state.cfg.checkpoint.load_optim = False

        # When
        bridge_state.build_load_state_dict(load_ctx)

        # Then
        assert_that(bridge_state.generate_state_dict.call_args.args[2]).is_none()

    def test_omits_the_optimizer_when_finetuning(self, load_ctx, mocker):
        # Given
        load_ctx.optimizer = mocker.MagicMock()
        load_ctx.state.cfg.checkpoint.finetune = True

        # When
        bridge_state.build_load_state_dict(load_ctx)

        # Then
        assert_that(bridge_state.generate_state_dict.call_args.args[2]).is_none()

    def test_includes_the_optimizer_on_a_plain_resume(self, load_ctx, mocker):
        # Given
        load_ctx.optimizer = mocker.MagicMock()

        # When
        bridge_state.build_load_state_dict(load_ctx)

        # Then
        assert_that(bridge_state.generate_state_dict.call_args.args[2]).is_same_as(load_ctx.optimizer)


class TestRestoreTrainState:
    def test_restores_embedded_counters(self, cfg):
        # Given
        state = FakeGlobalState(cfg=cfg)
        state_dict = {bridge_state.TRAIN_STATE_KEY: {"step": 77, "consumed_train_samples": 1024}}

        # When
        bridge_state.restore_train_state(state, state_dict)

        # Then
        assert_that(state.train_state.step).is_equal_to(77)
        assert_that(state.train_state.consumed_train_samples).is_equal_to(1024)

    def test_falls_back_to_the_iteration_key(self, cfg):
        # Given
        state = FakeGlobalState(cfg=cfg)

        # When
        bridge_state.restore_train_state(state, {"iteration": 12})

        # Then
        assert_that(state.train_state.step).is_equal_to(12)

    def test_restores_flops(self, cfg):
        # Given
        state = FakeGlobalState(cfg=cfg)

        # When
        bridge_state.restore_train_state(state, {"iteration": 1, bridge_state.FLOPS_KEY: 4321})

        # Then
        assert_that(state.train_state.floating_point_operations_so_far).is_equal_to(4321)


class TestRestoreModel:
    def test_single_chunk(self, mocker):
        # Given
        loader = mocker.MagicMock()
        mocker.patch.object(bridge_state, "_import_optional", return_value=loader)
        model = [object()]

        # When
        bridge_state.restore_model(model, {"model": {"w": 1}}, strict=True)

        # Then
        loader.assert_called_once_with(model[0], {"w": 1}, True)

    def test_multiple_chunks(self, mocker):
        # Given
        loader = mocker.MagicMock()
        mocker.patch.object(bridge_state, "_import_optional", return_value=loader)
        model = [object(), object()]

        # When
        bridge_state.restore_model(model, {"model0": {"a": 1}, "model1": {"b": 2}}, strict=False)

        # Then
        assert_that(loader.call_count).is_equal_to(2)

    def test_skips_empty_pipeline_stages(self, mocker):
        # Given
        loader = mocker.MagicMock()
        mocker.patch.object(bridge_state, "_import_optional", return_value=loader)
        model = [object(), object()]

        # When
        bridge_state.restore_model(model, {"model0": {"a": 1}}, strict=True)

        # Then
        assert_that(loader.call_count).is_equal_to(1)

    def test_raises_when_the_bridge_helper_is_missing(self, mocker):
        # Given
        mocker.patch.object(bridge_state, "_import_optional", return_value=None)

        # When/Then
        with pytest.raises(RuntimeError, match="_load_model_state_dict"):
            bridge_state.restore_model([object()], {"model": {}}, strict=True)


class TestRestoreOptimizer:
    def test_restores_optimizer_and_scheduler(self, load_ctx, mocker):
        # Given
        load_ctx.optimizer = mocker.MagicMock(is_stub_optimizer=False)
        load_ctx.opt_param_scheduler = mocker.MagicMock()

        # When
        bridge_state.restore_optimizer(load_ctx, {"optimizer": {"o": 1}, "opt_param_scheduler": {"s": 2}})

        # Then
        load_ctx.optimizer.load_state_dict.assert_called_once_with({"o": 1})
        load_ctx.opt_param_scheduler.load_state_dict.assert_called_once_with({"s": 2})

    def test_prefers_the_legacy_lr_scheduler_key(self, load_ctx, mocker):
        # Given
        load_ctx.opt_param_scheduler = mocker.MagicMock()

        # When
        bridge_state.restore_optimizer(load_ctx, {"lr_scheduler": {"legacy": True}})

        # Then
        load_ctx.opt_param_scheduler.load_state_dict.assert_called_once_with({"legacy": True})

    def test_skips_stub_optimizers(self, load_ctx, mocker):
        # Given
        load_ctx.optimizer = mocker.MagicMock(is_stub_optimizer=True)

        # When
        bridge_state.restore_optimizer(load_ctx, {"optimizer": {"o": 1}})

        # Then
        load_ctx.optimizer.load_state_dict.assert_not_called()

    def test_skips_everything_when_finetuning(self, load_ctx, mocker):
        # Given
        load_ctx.optimizer = mocker.MagicMock(is_stub_optimizer=False)
        load_ctx.opt_param_scheduler = mocker.MagicMock()
        load_ctx.state.cfg.checkpoint.finetune = True

        # When
        bridge_state.restore_optimizer(load_ctx, {"optimizer": {}, "opt_param_scheduler": {}})

        # Then
        load_ctx.optimizer.load_state_dict.assert_not_called()
        load_ctx.opt_param_scheduler.load_state_dict.assert_not_called()

    def test_skips_the_optimizer_when_skipping_model_load(self, load_ctx, mocker):
        # Given
        load_ctx.optimizer = mocker.MagicMock(is_stub_optimizer=False)
        load_ctx.skip_load_to_model_and_opt = True

        # When
        bridge_state.restore_optimizer(load_ctx, {"optimizer": {"o": 1}})

        # Then
        load_ctx.optimizer.load_state_dict.assert_not_called()


class TestRestoreRerunState:
    def test_restores_when_present(self, mocker):
        # Given
        machine = mocker.MagicMock()
        mocker.patch.object(bridge_state, "get_rerun_state_machine", return_value=machine)

        # When
        bridge_state.restore_rerun_state({"rerun_state_machine": {"x": 1}})

        # Then
        machine.load_state_dict.assert_called_once_with({"x": 1})

    def test_absent_key_is_a_noop(self, mocker):
        # Given
        machine = mocker.MagicMock()
        mocker.patch.object(bridge_state, "get_rerun_state_machine", return_value=machine)

        # When
        bridge_state.restore_rerun_state({})

        # Then
        machine.load_state_dict.assert_not_called()

    def test_failure_is_swallowed(self, mocker):
        # Given
        machine = mocker.MagicMock()
        machine.load_state_dict.side_effect = RuntimeError("incompatible")
        mocker.patch.object(bridge_state, "get_rerun_state_machine", return_value=machine)

        # When
        bridge_state.restore_rerun_state({"rerun_state_machine": {}})

        # Then no exception escapes.


class TestRestoreRngState:
    def _rng_state(self):
        return {
            "random_rng_state": __import__("random").getstate(),
            "np_rng_state": __import__("numpy").random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.get_rng_state(),
            "rng_tracker_states": {"tracker": "state"},
        }

    def test_restores_the_data_parallel_rank_slot(self, load_ctx, pg_collection, mocker):
        # Given
        load_ctx.state.cfg.rng.data_parallel_random_init = True
        pg_collection.dp.rank.return_value = 1
        tracker = mocker.MagicMock()
        mocker.patch.object(bridge_state.tensor_parallel, "get_cuda_rng_tracker", return_value=tracker)
        mocker.patch.object(
            bridge_state.tensor_parallel, "is_graph_safe_cuda_rng_tracker", return_value=False, create=True
        )
        mocker.patch.object(
            bridge_state.tensor_parallel, "convert_cuda_rng_state", side_effect=lambda v, **_: v, create=True
        )
        mocker.patch.object(torch.cuda, "set_rng_state")
        state_dict = {"rng_state": [self._rng_state(), self._rng_state()]}

        # When
        bridge_state.restore_rng_state(load_ctx, state_dict, pg_collection)

        # Then
        tracker.set_states.assert_called_once_with({"tracker": "state"})

    def test_skips_when_load_rng_is_off(self, load_ctx, pg_collection, mocker):
        # Given
        load_ctx.state.cfg.checkpoint.load_rng = False
        tracker = mocker.MagicMock()
        mocker.patch.object(bridge_state.tensor_parallel, "get_cuda_rng_tracker", return_value=tracker)

        # When
        bridge_state.restore_rng_state(load_ctx, {"rng_state": [self._rng_state()]}, pg_collection)

        # Then
        tracker.set_states.assert_not_called()

    def test_missing_rng_state_is_a_noop(self, load_ctx, pg_collection, mocker):
        # Given
        tracker = mocker.MagicMock()
        mocker.patch.object(bridge_state.tensor_parallel, "get_cuda_rng_tracker", return_value=tracker)

        # When
        bridge_state.restore_rng_state(load_ctx, {}, pg_collection)

        # Then
        tracker.set_states.assert_not_called()

    def test_failure_is_swallowed(self, load_ctx, pg_collection, mocker):
        # Given a payload missing the keys the restore path needs.
        mocker.patch.object(bridge_state.tensor_parallel, "get_cuda_rng_tracker", return_value=mocker.MagicMock())

        # When
        bridge_state.restore_rng_state(load_ctx, {"rng_state": [{}]}, pg_collection)

        # Then no exception escapes.


class TestApplyLoadedState:
    def test_returns_step_and_flops(self, load_ctx, mocker):
        # Given
        mocker.patch.object(bridge_state, "set_checkpoint_version")
        mocker.patch.object(bridge_state, "update_num_microbatches")
        mocker.patch.object(bridge_state, "restore_model")
        mocker.patch.object(bridge_state, "restore_optimizer")
        mocker.patch.object(bridge_state, "restore_rerun_state")
        mocker.patch.object(bridge_state, "restore_rng_state")
        state_dict = {bridge_state.TRAIN_STATE_KEY: {"step": 88}, bridge_state.FLOPS_KEY: 2048}

        # When
        step, flops = bridge_state.apply_loaded_state(load_ctx, state_dict)

        # Then
        assert_that(step).is_equal_to(88)
        assert_that(flops).is_equal_to(2048)

    def test_skips_the_model_when_asked_to(self, load_ctx, mocker):
        # Given
        mocker.patch.object(bridge_state, "set_checkpoint_version")
        mocker.patch.object(bridge_state, "update_num_microbatches")
        restore_model = mocker.patch.object(bridge_state, "restore_model")
        mocker.patch.object(bridge_state, "restore_optimizer")
        mocker.patch.object(bridge_state, "restore_rerun_state")
        mocker.patch.object(bridge_state, "restore_rng_state")
        load_ctx.skip_load_to_model_and_opt = True

        # When
        bridge_state.apply_loaded_state(load_ctx, {bridge_state.TRAIN_STATE_KEY: {"step": 1}})

        # Then
        restore_model.assert_not_called()


class TestMisc:
    def test_get_content_metadata(self):
        # Given/When/Then
        assert_that(bridge_state.get_content_metadata({bridge_state.CONTENT_METADATA_KEY: {"v": 1}})).is_equal_to(
            {"v": 1}
        )
        assert_that(bridge_state.get_content_metadata({})).is_none()

    def test_resolve_pg_collection_prefers_the_explicit_value(self, mocker):
        # Given
        explicit = mocker.MagicMock()
        get_pg = mocker.patch.object(bridge_state, "get_pg_collection")

        # When
        result = bridge_state.resolve_pg_collection([object()], explicit)

        # Then
        assert_that(result).is_same_as(explicit)
        get_pg.assert_not_called()

    def test_resolve_pg_collection_falls_back_to_the_model(self, mocker):
        # Given
        derived = mocker.MagicMock()
        mocker.patch.object(bridge_state, "get_pg_collection", return_value=derived)

        # When
        result = bridge_state.resolve_pg_collection([object()], None)

        # Then
        assert_that(result).is_same_as(derived)

    def test_build_sharded_state_dict_metadata_requires_the_bridge_helper(self, mocker):
        # Given
        mocker.stopall()
        mocker.patch.object(bridge_state, "_import_optional", return_value=None)

        # When/Then
        with pytest.raises(RuntimeError, match="_build_sharded_state_dict_metadata"):
            bridge_state.build_sharded_state_dict_metadata(True, object())

    def test_clean_metadata_drops_process_groups_without_the_helper(self, mocker):
        # Given a Megatron build that predates the serialization helper.
        real_import = bridge_state.importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == "megatron.core.dist_checkpointing.utils":
                raise ImportError("not available")
            return real_import(name, *args, **kwargs)

        mocker.patch.object(bridge_state.importlib, "import_module", side_effect=fake_import)

        # When
        result = bridge_state.clean_metadata_for_serialization({"keep": 1})

        # Then
        assert_that(result).is_equal_to({"keep": 1})
