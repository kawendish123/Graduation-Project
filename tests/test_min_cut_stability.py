import networkx as nx

from dads_dsl.dsl import (
    SINK,
    SOURCE,
    _directed_cut_capacity,
    _integer_capacity_graph,
    _minimum_cut_partition,
    construct_flow_graph,
    solve_dsl,
    with_virtual_input_node,
)
from dads_dsl.types import ModelProfile, ModelProfileNode


def _profile_for_bandwidth_choice() -> ModelProfile:
    return ModelProfile(
        model_name="toy",
        input_shape=[1, 3, 224, 224],
        nodes=[
            ModelProfileNode("a", "Conv2d", ["b"], 100.0, 1.0, 602_112),
            ModelProfileNode("b", "Conv2d", [], 100.0, 1.0, 4),
        ],
    )


def test_integer_minimum_cut_partition_matches_cut_value():
    profile = with_virtual_input_node(_profile_for_bandwidth_choice(), input_bytes=602_112)
    build = construct_flow_graph(profile, bandwidth_mbps=5.0)
    scaled_graph = _integer_capacity_graph(build.graph)

    cut_value, _ = nx.minimum_cut(scaled_graph, SOURCE, SINK, capacity="capacity")
    reachable, non_reachable = _minimum_cut_partition(build.graph)

    assert SOURCE in reachable
    assert SINK in non_reachable
    assert _directed_cut_capacity(scaled_graph, reachable, non_reachable) == cut_value


def test_low_bandwidth_virtual_input_does_not_beat_all_edge_with_expensive_transfer():
    profile = with_virtual_input_node(_profile_for_bandwidth_choice(), input_bytes=602_112)
    solution = solve_dsl(profile, bandwidth_mbps=5.0)
    all_edge_estimated_ms = sum(node.edge_ms for node in profile.nodes)

    assert solution.cloud_nodes == []
    assert solution.transmission_nodes == []
    assert solution.total_inference_ms == all_edge_estimated_ms


def test_high_bandwidth_can_choose_cloud_with_virtual_input_transfer():
    profile = with_virtual_input_node(_profile_for_bandwidth_choice(), input_bytes=4)
    solution = solve_dsl(profile, bandwidth_mbps=1_000_000.0)

    assert solution.edge_nodes == ["__input__"]
    assert solution.transmission_nodes == ["__input__"]
    assert solution.cloud_nodes == ["a", "b"]


def test_solution_does_not_create_cloud_to_edge_dependencies():
    profile = ModelProfile(
        model_name="roundtrip",
        input_shape=[1, 3, 4, 4],
        nodes=[
            ModelProfileNode("a", "Conv2d", ["b"], 100.0, 1.0, 4),
            ModelProfileNode("b", "ReLU", ["c"], 1.0, 100.0, 4),
            ModelProfileNode("c", "Linear", [], 100.0, 1.0, 4),
        ],
    )
    solution = solve_dsl(with_virtual_input_node(profile, input_bytes=4), bandwidth_mbps=1_000_000.0)
    edge_nodes = set(solution.edge_nodes)
    cloud_nodes = set(solution.cloud_nodes)

    for node in profile.nodes:
        for succ_id in node.succ_ids:
            assert not (node.id in cloud_nodes and succ_id in edge_nodes)
