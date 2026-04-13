from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelProfileNode:
    id: str
    op_type: str
    succ_ids: list[str]
    edge_ms: float
    cloud_ms: float
    output_bytes: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProfileNode":
        return cls(
            id=str(data["id"]),
            op_type=str(data["op_type"]),
            succ_ids=[str(item) for item in data.get("succ_ids", [])],
            edge_ms=float(data["edge_ms"]),
            cloud_ms=float(data["cloud_ms"]),
            output_bytes=int(data["output_bytes"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelProfile:
    model_name: str
    input_shape: list[int]
    nodes: list[ModelProfileNode]
    metadata: Optional[dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelProfile":
        nodes = [ModelProfileNode.from_dict(node) for node in data["nodes"]]
        return cls(
            model_name=str(data["model_name"]),
            input_shape=[int(item) for item in data["input_shape"]],
            nodes=nodes,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ModelProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "input_shape": self.input_shape,
            "nodes": [node.to_dict() for node in self.nodes],
            "metadata": self.metadata or {},
        }

    def node_map(self) -> dict[str, ModelProfileNode]:
        return {node.id: node for node in self.nodes}


@dataclass
class DSLSolution:
    model_name: str
    bandwidth_mbps: float
    total_inference_ms: float
    edge_stage_ms: float
    transmission_stage_ms: float
    cloud_stage_ms: float
    edge_nodes: list[str]
    transmission_nodes: list[str]
    cloud_nodes: list[str]
    cut_edges: list[tuple[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "bandwidth_mbps": self.bandwidth_mbps,
            "total_inference_ms": self.total_inference_ms,
            "edge_stage_ms": self.edge_stage_ms,
            "transmission_stage_ms": self.transmission_stage_ms,
            "cloud_stage_ms": self.cloud_stage_ms,
            "edge_nodes": self.edge_nodes,
            "transmission_nodes": self.transmission_nodes,
            "cloud_nodes": self.cloud_nodes,
            "cut_edges": [list(edge) for edge in self.cut_edges],
        }
