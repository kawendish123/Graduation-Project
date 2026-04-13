# %%

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import networkx as nx

from .types import DSLSolution, ModelProfile, ModelProfileNode

SOURCE = "__source__"
SINK = "__sink__"
VIRTUAL_INPUT_ID = "__input__"
CAPACITY_SCALE = 1_000_000


def with_virtual_input_node(profile: ModelProfile, input_bytes: int) -> ModelProfile:
    if input_bytes < 0:
        raise ValueError("input_bytes must be non-negative.")
    if any(node.id == VIRTUAL_INPUT_ID for node in profile.nodes):
        return profile

    all_successors = {succ_id for node in profile.nodes for succ_id in node.succ_ids}
    root_ids = [node.id for node in profile.nodes if node.id not in all_successors]
    virtual_input_node = ModelProfileNode(
        id=VIRTUAL_INPUT_ID,
        op_type="Input",
        succ_ids=root_ids,
        edge_ms=0.0,
        cloud_ms=0.0,
        output_bytes=input_bytes,
    )
    return ModelProfile(
        model_name=profile.model_name,
        input_shape=list(profile.input_shape),
        nodes=[virtual_input_node] + list(profile.nodes),
        metadata=profile.metadata,
    )


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


def _finite_capacity_sum(profile: ModelProfile, transmission_ms: dict[str, float]) -> float:
    total = 0.0
    for node in profile.nodes:
        total += node.edge_ms + node.cloud_ms
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
            graph.add_edge(
                node.succ_ids[0],
                node.id,
                capacity=infinity_capacity,
                role="precedence_constraint",
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
                graph.add_edge(
                    succ_id,
                    node.id,
                    capacity=infinity_capacity,
                    role="precedence_constraint",
                )

    return FlowGraphBuild(
        graph=graph,
        transmission_ms=transmission_ms,
        aux_nodes=aux_nodes,
        infinity_capacity=infinity_capacity,
    )


def _scaled_capacity(capacity: float) -> int:
    if not math.isfinite(capacity):
        raise ValueError("All graph capacities must be finite.")
    if capacity < 0:
        raise ValueError("All graph capacities must be non-negative.")
    if capacity == 0:
        return 0
    return max(1, int(round(capacity * CAPACITY_SCALE)))


def _integer_capacity_graph(graph: nx.DiGraph) -> nx.DiGraph:
    scaled_graph = nx.DiGraph()
    scaled_graph.add_nodes_from(graph.nodes(data=True))
    for src, dst, data in graph.edges(data=True):
        copied = dict(data)
        copied["capacity"] = _scaled_capacity(float(data["capacity"]))
        scaled_graph.add_edge(src, dst, **copied)
    return scaled_graph


def _directed_cut_capacity(graph: nx.DiGraph, reachable: set[str], non_reachable: set[str]) -> int:
    return sum(
        int(graph[src][dst]["capacity"])
        for src in reachable
        for dst in graph.successors(src)
        if dst in non_reachable
    )


def _minimum_cut_partition(graph: nx.DiGraph) -> tuple[set[str], set[str]]:
    scaled_graph = _integer_capacity_graph(graph)
    cut_value, partition = nx.minimum_cut(scaled_graph, SOURCE, SINK, capacity="capacity")
    reachable, non_reachable = partition

    if SOURCE not in reachable or SINK not in non_reachable:
        raise RuntimeError("Minimum cut partition is invalid: source and sink are not separated.")

    directed_cut_value = _directed_cut_capacity(scaled_graph, reachable, non_reachable)
    if directed_cut_value != cut_value:
        raise RuntimeError(
            "Minimum cut partition is inconsistent with the cut value: "
            f"partition_capacity={directed_cut_value}, cut_value={cut_value}."
        )

    return set(reachable), set(non_reachable)


def _sorted_nodes(nodes: Iterable[str], order: dict[str, int]) -> list[str]:
    return sorted(nodes, key=lambda item: order[item])


def solve_dsl(profile: ModelProfile, bandwidth_mbps: float) -> DSLSolution:
    build = construct_flow_graph(profile, bandwidth_mbps)
    graph = build.graph
    reachable, non_reachable = _minimum_cut_partition(graph)

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


def build_debug_payload(profile: ModelProfile, bandwidth_mbps: float, solution: DSLSolution) -> dict:
    build = construct_flow_graph(profile, bandwidth_mbps)
    transmission_ms = build.transmission_ms
    edge_node_ids = set(solution.edge_nodes)
    cloud_node_ids = set(solution.cloud_nodes)
    transmission_node_ids = set(solution.transmission_nodes)
    cut_edges = set(tuple(edge) for edge in solution.cut_edges)

    nodes = []
    for node in profile.nodes:
        if node.id in edge_node_ids:
            partition = "edge"
        elif node.id in cloud_node_ids:
            partition = "cloud"
        else:
            partition = "unknown"
        nodes.append(
            {
                "id": node.id,
                "op_type": node.op_type,
                "succ_ids": list(node.succ_ids),
                "edge_ms": node.edge_ms,
                "cloud_ms": node.cloud_ms,
                "transmission_ms": transmission_ms[node.id],
                "output_bytes": node.output_bytes,
                "partition": partition,
                "is_transmission_node": node.id in transmission_node_ids,
            }
        )

    flow_edges = []
    for src, dst, data in sorted(build.graph.edges(data=True), key=lambda item: (str(item[0]), str(item[1]))):
        flow_edges.append(
            {
                "src": src,
                "dst": dst,
                "capacity_ms": float(data["capacity"]),
                "role": data.get("role", ""),
                "is_cut_edge": (src, dst) in cut_edges,
            }
        )

    all_edge_estimated_ms = sum(node.edge_ms for node in profile.nodes)
    return {
        "model_name": profile.model_name,
        "input_shape": list(profile.input_shape),
        "bandwidth_mbps": bandwidth_mbps,
        "summary": {
            "edge_node_count": len(solution.edge_nodes),
            "cloud_node_count": len(solution.cloud_nodes),
            "transmission_node_count": len(solution.transmission_nodes),
            "dsl_estimated_edge_ms": solution.edge_stage_ms,
            "dsl_estimated_transfer_ms": solution.transmission_stage_ms,
            "dsl_estimated_cloud_ms": solution.cloud_stage_ms,
            "dsl_estimated_total_ms": solution.total_inference_ms,
            "all_edge_estimated_ms": all_edge_estimated_ms,
        },
        "partition": solution.to_dict(),
        "nodes": nodes,
        "flow_edges": flow_edges,
        "cut_edges": [list(edge) for edge in solution.cut_edges],
        "aux_nodes": dict(build.aux_nodes),
    }


def solve_bandwidth_sweep(profile: ModelProfile, bandwidths_mbps: Iterable[float]) -> list[DSLSolution]:
    cleaned = []
    for bandwidth in bandwidths_mbps:
        if not math.isfinite(bandwidth):
            raise ValueError("All bandwidth values must be finite.")
        cleaned.append(float(bandwidth))
    return [solve_dsl(profile, bandwidth) for bandwidth in cleaned]
