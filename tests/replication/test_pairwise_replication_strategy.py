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

from ml_flashpoint.replication.replication_manager import PairwiseReplicationStrategy


@pytest.fixture
def mock_gpus_per_node_2(mocker):
    mocker.patch("torch.cuda.device_count", return_value=2)
    return 2


@pytest.fixture(autouse=True)
def mock_num_nodes(mocker):
    mocker.patch("ml_flashpoint.core.utils.get_num_of_nodes", return_value=4)
    return 4


@pytest.fixture
def mock_gpus_per_node_8(mocker):
    mocker.patch("torch.cuda.device_count", return_value=8)
    return 8


@pytest.fixture
def replication_service_addresses_2_gpus():
    # 4 nodes, 2 GPUs each = 8 ranks
    return [
        "addr_n0_g0",
        "addr_n0_g1",  # Node 0
        "addr_n1_g0",
        "addr_n1_g1",  # Node 1
        "addr_n2_g0",
        "addr_n2_g1",  # Node 2
        "addr_n3_g0",
        "addr_n3_g1",  # Node 3
    ]


@pytest.fixture
def replication_service_addresses_8_gpus():
    # 4 nodes, 8 GPUs each = 32 ranks
    view = []
    for n in range(4):
        for g in range(8):
            view.append(f"addr_n{n}_g{g}")
    return view


def test_initialization_valid(mock_gpus_per_node_2, replication_service_addresses_2_gpus):
    strategy = PairwiseReplicationStrategy(replication_service_addresses_2_gpus, processes_per_node=2)
    assert strategy._processes_per_node == 2
    assert strategy._world_size == 8
    assert strategy._num_nodes == 4


def test_initialization_empty_view(mock_gpus_per_node_2):
    with pytest.raises(ValueError, match="The replication_service_addresses cannot be empty."):
        PairwiseReplicationStrategy([], processes_per_node=2)


def test_initialization_none_view(mock_gpus_per_node_2):
    with pytest.raises(ValueError, match="The replication_service_addresses cannot be empty."):
        PairwiseReplicationStrategy(None, processes_per_node=2)


def test_initialization_odd_nodes(mock_gpus_per_node_2, mocker):
    # 3 nodes * 2 GPUs = 6 ranks
    mocker.patch("ml_flashpoint.core.utils.get_num_of_nodes", return_value=3)
    odd_view = ["addr"] * 6
    with pytest.raises(ValueError, match="The total number of nodes .* must be even."):
        PairwiseReplicationStrategy(odd_view, processes_per_node=2)


def test_initialization_world_size_mismatch(mock_gpus_per_node_2):
    # 5 ranks, 2 GPUs per node -> Invalid
    invalid_view = ["addr"] * 5
    with pytest.raises(ValueError, match="World size .* must be divisible by processes per node"):
        PairwiseReplicationStrategy(invalid_view, processes_per_node=2)


@pytest.mark.parametrize(
    "rank, expected_dest",
    [
        (0, ["addr_n1_g0"]),
        (1, ["addr_n1_g1"]),
        (2, ["addr_n0_g0"]),
        (5, ["addr_n3_g1"]),
        (6, ["addr_n2_g0"]),
    ],
)
def test_get_destination_addresses_2_gpus(
    mock_gpus_per_node_2, replication_service_addresses_2_gpus, rank, expected_dest
):
    strategy = PairwiseReplicationStrategy(replication_service_addresses_2_gpus, processes_per_node=2)
    assert strategy.get_destination_addresses(rank) == expected_dest


@pytest.mark.parametrize(
    "rank, expected_dest",
    [
        (0, ["addr_n1_g0"]),
        (7, ["addr_n1_g7"]),
        (8, ["addr_n0_g0"]),
        (16, ["addr_n3_g0"]),
        (31, ["addr_n2_g7"]),
    ],
)
def test_get_destination_addresses_8_gpus(
    mock_gpus_per_node_8, replication_service_addresses_8_gpus, rank, expected_dest
):
    strategy = PairwiseReplicationStrategy(replication_service_addresses_8_gpus, processes_per_node=8)
    assert strategy.get_destination_addresses(rank) == expected_dest


@pytest.mark.parametrize("invalid_rank", [-1, 8])
def test_get_destination_addresses_invalid_rank(
    mock_gpus_per_node_2, replication_service_addresses_2_gpus, invalid_rank
):
    strategy = PairwiseReplicationStrategy(replication_service_addresses_2_gpus, processes_per_node=2)
    with pytest.raises(ValueError, match="out of valid range"):
        strategy.get_destination_addresses(invalid_rank)
