from __future__ import annotations

from itertools import product

from dads_dsl.dsl import solve_dsl
from dads_dsl.types import ModelProfile, ModelProfileNode


def _chain_profile() -> ModelProfile:
    return ModelProfile(
        model_name="chain",
        input_shape=[1, 3, 4, 4],
        nodes=[
            ModelProfileNode("a", "Conv2d", ["b"], 1.0, 9.0, 125_000),
            ModelProfileNode("b", "Conv2d", ["c"], 3.0, 4.0, 125_000),
            ModelProfileNode("c", "Linear", [], 8.0, 1.0, 32),
        ],
    )


def _branch_profile() -> ModelProfile:
    return ModelProfile(
        model_name="branch",
        input_shape=[1, 3, 4, 4],
        nodes=[
            ModelProfileNode("a", "Conv2d", ["b", "c"], 2.0, 7.0, 125_000),
            ModelProfileNode("b", "Conv2d", ["d"], 2.0, 7.0, 32),
            ModelProfileNode("c", "Conv2d", ["d"], 2.0, 7.0, 32),
            ModelProfileNode("d", "Linear", [], 7.0, 1.0, 16),
        ],
    )


def _bruteforce_total(profile: ModelProfile, bandwidth_mbps: float):
    node_map = profile.node_map()
    ids = [node.id for node in profile.nodes]
    predecessors = {node.id: set() for node in profile.nodes}
    for node in profile.nodes:
        for succ_id in node.succ_ids:
            predecessors[succ_id].add(node.id)

    best = None
    best_assignment = None
    for choices in product([False, True], repeat=len(ids)):
        edge_nodes = {ids[index] for index, is_edge in enumerate(choices) if is_edge}
        if any(not predecessors[node_id].issubset(edge_nodes) for node_id in edge_nodes):
            continue
        cloud_nodes = set(ids) - edge_nodes
        transmission_nodes = {
            node_id
            for node_id in edge_nodes
            if any(succ_id in cloud_nodes for succ_id in node_map[node_id].succ_ids)
        }
        edge_stage_ms = sum(node_map[node_id].edge_ms for node_id in edge_nodes)
        tx_stage_ms = sum((node_map[node_id].output_bytes * 8.0 / (bandwidth_mbps * 1_000_000.0)) * 1000.0 for node_id in transmission_nodes)
        cloud_stage_ms = sum(node_map[node_id].cloud_ms for node_id in cloud_nodes)
        total = edge_stage_ms + tx_stage_ms + cloud_stage_ms
        if best is None or total < best:
            best = total
            best_assignment = (
                sorted(edge_nodes),
                sorted(transmission_nodes),
                sorted(cloud_nodes),
            )
    return best, best_assignment


def test_chain_solution_matches_bruteforce():
    profile = _chain_profile()
    expected_total, expected_sets = _bruteforce_total(profile, bandwidth_mbps=1.0)
    solution = solve_dsl(profile, bandwidth_mbps=1.0)
    assert solution.total_inference_ms == expected_total
    assert (
        solution.edge_nodes,
        solution.transmission_nodes,
        solution.cloud_nodes,
    ) == expected_sets


def test_branch_solution_matches_bruteforce():
    profile = _branch_profile()
    expected_total, expected_sets = _bruteforce_total(profile, bandwidth_mbps=1.0)
    solution = solve_dsl(profile, bandwidth_mbps=1.0)
    assert solution.total_inference_ms == expected_total
    assert (
        solution.edge_nodes,
        solution.transmission_nodes,
        solution.cloud_nodes,
    ) == expected_sets
