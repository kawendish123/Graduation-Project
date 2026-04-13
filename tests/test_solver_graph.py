from dads_dsl.dsl import SOURCE, SINK, construct_flow_graph, solve_dsl
from dads_dsl.types import ModelProfile, ModelProfileNode


def _branch_profile() -> ModelProfile:
    return ModelProfile(
        model_name="branch",
        input_shape=[1, 3, 4, 4],
        nodes=[
            ModelProfileNode("a", "Conv2d", ["b", "c"], 1.0, 100.0, 1_000),
            ModelProfileNode("b", "Conv2d", ["d"], 100.0, 1.0, 500),
            ModelProfileNode("c", "Conv2d", ["d"], 100.0, 1.0, 500),
            ModelProfileNode("d", "Conv2d", [], 100.0, 1.0, 250),
        ],
    )


def test_construct_flow_graph_adds_aux_node_for_multi_successor_vertex():
    build = construct_flow_graph(_branch_profile(), bandwidth_mbps=1.0)
    assert "a__aux" in build.graph.nodes
    assert build.graph["a"]["a__aux"]["role"] == "transmission"
    assert build.graph["a"]["a__aux"]["capacity"] == build.transmission_ms["a"]
    assert build.graph["a__aux"]["b"]["capacity"] == build.infinity_capacity
    assert build.graph["a__aux"]["c"]["capacity"] == build.infinity_capacity


def test_construct_flow_graph_adds_source_and_sink_edges():
    build = construct_flow_graph(_branch_profile(), bandwidth_mbps=2.0)
    assert build.graph[SOURCE]["a"]["capacity"] == 100.0
    assert build.graph["d"][SINK]["capacity"] == 100.0


def test_solution_marks_transmission_node_once_even_with_two_successors():
    solution = solve_dsl(_branch_profile(), bandwidth_mbps=1.0)
    assert solution.transmission_nodes == ["a"]
    assert solution.transmission_stage_ms == 8.0
