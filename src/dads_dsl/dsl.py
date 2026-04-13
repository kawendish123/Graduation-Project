from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import networkx as nx

from .types import DSLSolution, ModelProfile
from .types import ModelProfileNode

SOURCE = "__source__"
SINK = "__sink__"
VIRTUAL_INPUT_ID = "__input__"


@dataclass
class FlowGraphBuild:
    graph: nx.DiGraph
    transmission_ms: dict[str, float]
    aux_nodes: dict[str, str]
    infinity_capacity: float


def compute_transmission_ms(output_bytes: int, bandwidth_mbps: float) -> float:
    if bandwidth_mbps <= 0:
        raise ValueError("bandwidth_mbps must be positive.")
    if output_bytes < 0:
        raise ValueError("output_bytes must be non-negative.")
    return (float(output_bytes) * 8.0 / (bandwidth_mbps * 1_000_000.0)) * 1000.0


def compute_transmission_profile(profile: ModelProfile, bandwidth_mbps: float) -> dict[str, float]:
    return {
        node.id: compute_transmission_ms(node.output_bytes, bandwidth_mbps)
        for node in profile.nodes
    }


def with_virtual_input_node(profile: ModelProfile, input_bytes: int) -> ModelProfile:
    if input_bytes < 0:
        raise ValueError("input_bytes must be non-negative.")
    if any(node.id == VIRTUAL_INPUT_ID for node in profile.nodes):
        return profile

    all_successors = {succ_id for node in profile.nodes for succ_id in node.succ_ids}
    root_nodes = [node.id for node in profile.nodes if node.id not in all_successors]
    virtual_input = ModelProfileNode(
        id=VIRTUAL_INPUT_ID,
        op_type="Input",
        succ_ids=root_nodes,
        edge_ms=0.0,
        cloud_ms=0.0,
        output_bytes=input_bytes,
    )
    metadata = dict(profile.metadata or {})
    metadata["virtual_input_node"] = VIRTUAL_INPUT_ID
    return ModelProfile(
        model_name=profile.model_name,
        input_shape=list(profile.input_shape),
        nodes=[virtual_input] + list(profile.nodes),
        metadata=metadata,
    )


def _finite_capacity_sum(profile: ModelProfile, transmission_ms: dict[str, float]) -> float:
    total = 0.0
    for node in profile.nodes:
        total += node.edge_ms
        if node.id != VIRTUAL_INPUT_ID:
            total += node.cloud_ms
        if node.succ_ids:
            total += transmission_ms[node.id]
    return total


def construct_flow_graph(profile: ModelProfile, bandwidth_mbps: float) -> FlowGraphBuild:
    graph = nx.DiGraph()
    graph.add_node(SOURCE)
    graph.add_node(SINK)

    transmission_ms = compute_transmission_profile(profile, bandwidth_mbps)
    infinity_capacity = _finite_capacity_sum(profile, transmission_ms) + 1.0
    aux_nodes: dict[str, str] = {}

    for node in profile.nodes:
        graph.add_node(node.id, kind="original")
        source_capacity = infinity_capacity if node.id == VIRTUAL_INPUT_ID else node.cloud_ms
        graph.add_edge(SOURCE, node.id, capacity=source_capacity, role="cloud")
        graph.add_edge(node.id, SINK, capacity=node.edge_ms, role="edge")

        if len(node.succ_ids) == 1:
            graph.add_edge(
                node.id,
                node.succ_ids[0],
                capacity=transmission_ms[node.id],
                role="transmission",
            )
        elif len(node.succ_ids) > 1:
            aux_id = f"{node.id}__aux"
            aux_nodes[node.id] = aux_id
            graph.add_node(aux_id, kind="aux")
            graph.add_edge(
                node.id,
                aux_id,
                capacity=transmission_ms[node.id],
                role="transmission",
            )
            for succ_id in node.succ_ids:
                graph.add_edge(aux_id, succ_id, capacity=infinity_capacity, role="aux")

    return FlowGraphBuild(
        graph=graph,
        transmission_ms=transmission_ms,
        aux_nodes=aux_nodes,
        infinity_capacity=infinity_capacity,
    )


def _sorted_nodes(nodes: Iterable[str], order: dict[str, int]) -> list[str]:
    return sorted(nodes, key=lambda item: order[item])


def solve_dsl(profile: ModelProfile, bandwidth_mbps: float) -> DSLSolution:
    build = construct_flow_graph(profile, bandwidth_mbps)
    graph = build.graph
    _, partition = nx.minimum_cut(graph, SOURCE, SINK, capacity="capacity")
    reachable, non_reachable = partition

    original_order = {node.id: index for index, node in enumerate(profile.nodes)}
    node_map = profile.node_map()

    edge_nodes = _sorted_nodes(
        (node_id for node_id in reachable if node_id in node_map),
        original_order,
    )
    cloud_nodes = _sorted_nodes(
        (node_id for node_id in non_reachable if node_id in node_map),
        original_order,
    )

    transmission_nodes: list[str] = []
    for node_id in edge_nodes:
        node = node_map[node_id]
        if not node.succ_ids:
            continue
        if len(node.succ_ids) == 1:
            if node.succ_ids[0] in non_reachable:
                transmission_nodes.append(node_id)
        else:
            aux_id = build.aux_nodes[node_id]
            if aux_id in non_reachable:
                transmission_nodes.append(node_id)

    cut_edges: list[tuple[str, str]] = []
    for src in reachable:
        for dst in graph.successors(src):
            if dst in non_reachable:
                cut_edges.append((src, dst))

    edge_stage_ms = sum(node_map[node_id].edge_ms for node_id in edge_nodes)
    transmission_stage_ms = sum(build.transmission_ms[node_id] for node_id in transmission_nodes)
    cloud_stage_ms = sum(node_map[node_id].cloud_ms for node_id in cloud_nodes)

    return DSLSolution(
        model_name=profile.model_name,
        bandwidth_mbps=bandwidth_mbps,
        total_inference_ms=edge_stage_ms + transmission_stage_ms + cloud_stage_ms,
        edge_stage_ms=edge_stage_ms,
        transmission_stage_ms=transmission_stage_ms,
        cloud_stage_ms=cloud_stage_ms,
        edge_nodes=edge_nodes,
        transmission_nodes=transmission_nodes,
        cloud_nodes=cloud_nodes,
        cut_edges=sorted(cut_edges),
    )


def solve_bandwidth_sweep(profile: ModelProfile, bandwidths_mbps: Iterable[float]) -> list[DSLSolution]:
    cleaned = []
    for bandwidth in bandwidths_mbps:
        if not math.isfinite(bandwidth):
            raise ValueError("All bandwidth values must be finite.")
        cleaned.append(float(bandwidth))
    return [solve_dsl(profile, bandwidth) for bandwidth in cleaned]
