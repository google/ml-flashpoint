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

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml_flashpoint.adapter.nemo.nemo_checkpoint_loader import NeMoMLFlashpointCheckpointLoader
from ml_flashpoint.checkpoint_object_manager.checkpoint_object_manager import CheckpointObjectManager
from ml_flashpoint.core.checkpoint_id_types import CheckpointContainerId, CheckpointObjectId
from ml_flashpoint.replication.replication_manager import ReplicationManager


class TestNeMoCheckpointLoaderContext:
    @pytest.fixture
    def _setup_mocks(self, mocker):
        self.mock_global_rank = MagicMock(return_value=0)
        self.mock_local_rank = MagicMock(return_value=0)
        self.mock_world_size = MagicMock(return_value=1)
        self.mock_all_gather = MagicMock()
        self.mock_broadcast = MagicMock()

    @pytest.fixture
    def loader(self, mocker, _setup_mocks):
        ckpt_manager = CheckpointObjectManager()
        repl_manager = MagicMock(spec=ReplicationManager)
        return NeMoMLFlashpointCheckpointLoader(
            checkpoint_object_manager=ckpt_manager,
            replication_manager=repl_manager,
            global_rank_getter=self.mock_global_rank,
            local_rank_getter=self.mock_local_rank,
            broadcast_object_list_func=self.mock_broadcast,
            all_gather_object_func=self.mock_all_gather,
            world_size_getter=self.mock_world_size,
            recover_context=True,
        )

    def test_get_checkpoint_objects_by_rank_finds_context(self, loader, mocker):
        """Test that get_checkpoint_objects_by_rank finds files in context/ dir when recover_context=True."""
        self.mock_world_size.return_value = 1
        self.mock_all_gather.side_effect = lambda obj_list, local_obj: obj_list.__setitem__(0, local_obj)

        container_path = "/tmp/ckpt/step-1"
        container_id = CheckpointContainerId(container_path)

        # Mock fs
        # base dir has "context" directory
        # context dir has "file1.txt"
        def mock_listdir(path):
            if str(path) == container_path:
                return ["context", "other_file"]
            if str(path) == str(Path(container_path) / "context"):
                return ["file1.txt", "file2.txt"]
            return []

        def mock_walk(path):
            if str(path) == str(Path(container_path) / "context"):
                # Simulation of:
                # context/
                #   file1.txt
                #   file2.txt
                #   subdir/
                #     file3.txt
                # Yields: (root, dirs, files)
                yield (str(path), ["subdir"], ["file1.txt", "file2.txt"])
                yield (str(Path(path) / "subdir"), [], ["file3.txt"])
            return []

        def mock_isdir(path):
            if str(path) == container_path:
                return True
            if str(path) == str(Path(container_path) / "context"):
                return True
            return False

        mocker.patch("os.walk", side_effect=mock_walk)
        mocker.patch("os.listdir", side_effect=mock_listdir)
        mocker.patch("pathlib.Path.is_dir", new=mock_isdir)

        result = loader.get_checkpoint_objects_by_rank(container_id)

        assert 0 in result
        objs = result[0]

        paths = [str(o.data) for o in objs]
        expected_context_file1 = str(Path(container_path) / "context" / "file1.txt")
        expected_context_file2 = str(Path(container_path) / "context" / "file2.txt")
        expected_nested_file3 = str(Path(container_path) / "context" / "subdir" / "file3.txt")
        assert expected_context_file1 in paths
        assert expected_context_file2 in paths
        assert expected_nested_file3 in paths

    def test_compute_retrieval_plan_includes_context_optimized(self, loader, mocker):
        """
        Test that _compute_retrieval_plan includes context files ONLY for local rank 0 on each node.
        Scenario:
          - World Size: 4
          - Nodes: 2 (Ranks 0,1 on Node 0; Ranks 2,3 on Node 1)
          - Rank 0 has context files.
          - Rank 2 needs context files (different node).
          - Rank 1, 3 do NOT need context retrieval (same node as 0, 2).
        """
        checkpoint = CheckpointContainerId("/tmp/ckpt/step-1")

        # Mock metadata read (empty storage data)
        mock_metadata = MagicMock()
        mock_metadata.storage_data = {}
        mocker.patch.object(loader, "read_metadata", return_value=mock_metadata)

        self.mock_world_size.return_value = 4
        mocker.patch("ml_flashpoint.core.checkpoint_loader.get_num_of_nodes", return_value=2)
        self.mock_global_rank.return_value = 0

        ctx_file = str(Path(checkpoint.data) / "context" / "file1.txt")
        nested_ctx_file = str(Path(checkpoint.data) / "context" / "subdir" / "file3.txt")
        common_pt = str(Path(checkpoint.data) / "common.pt")
        metadata_file = str(Path(checkpoint.data) / ".metadata")

        # Available objects:
        # Node 0 (Rank 0,1) has everything (Context + Nested + Common + Metadata)
        # Node 1 (Rank 2,3) has nothing
        available_objects = {
            0: [
                CheckpointObjectId(ctx_file),
                CheckpointObjectId(nested_ctx_file),
                CheckpointObjectId(common_pt),
                CheckpointObjectId(metadata_file),
            ],
            1: [
                CheckpointObjectId(ctx_file),
                CheckpointObjectId(nested_ctx_file),
                CheckpointObjectId(common_pt),
                CheckpointObjectId(metadata_file),
            ],
            2: [],
            3: [],
        }

        # Execute
        plan = loader._compute_retrieval_plan(checkpoint, available_objects)

        assert plan is not None

        # Node 0: Already has files, no retrieval needed (or plan[0] is empty)
        assert 0 not in plan or not plan[0]
        assert 1 not in plan or not plan[1]

        # Rank 2: Local rank 0 on Node 1. Needs Context + Common + Metadata.
        assert 2 in plan
        retrieved_objs_2 = [path for src, path in plan[2]]
        assert ctx_file in retrieved_objs_2
        assert common_pt in retrieved_objs_2
        assert metadata_file in retrieved_objs_2

        # Verify nested file
        nested_ctx_file = str(Path(checkpoint.data) / "context" / "subdir" / "file3.txt")
        assert nested_ctx_file in retrieved_objs_2

        # Rank 3: Local rank 1 on Node 1. Shared FS with Rank 2. Should NOT retrieve.
        assert 3 not in plan or not plan[3]
