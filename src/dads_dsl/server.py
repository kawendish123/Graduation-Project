from __future__ import annotations

from concurrent import futures
from typing import Any

import grpc

from .profile import ProfileRunConfig, profile_model
from .rpc import add_dads_cloud_servicer_to_server, grpc_options
from .runtime import build_runtime_model, execute_cloud_partition
from .tensor_codec import payload_to_tensor
from .types import ModelProfile


class CloudRuntimeCache:
    def __init__(self, device: str):
        self.device = device
        self._profiles: dict[str, ModelProfile] = {}
        self._runtimes: dict[str, Any] = {}

    def get_profile(self, model_name: str, input_shape: list[int], warmup_runs: int, profile_runs: int) -> ModelProfile:
        key = f"{model_name}:{tuple(input_shape)}:{self.device}:{warmup_runs}:{profile_runs}"
        if key not in self._profiles:
            self._profiles[key] = profile_model(
                model_name,
                ProfileRunConfig(
                    input_shape=tuple(input_shape),
                    edge_warmup_runs=0,
                    edge_profile_runs=1,
                    cloud_warmup_runs=warmup_runs,
                    cloud_profile_runs=profile_runs,
                    cloud_device=self.device,
                ),
            )
        return self._profiles[key]

    def get_runtime(self, model_name: str, profile_node_ids: set[str]) -> Any:
        key = f"{model_name}:{self.device}"
        if key not in self._runtimes:
            self._runtimes[key] = build_runtime_model(model_name, self.device, profile_node_ids)
        return self._runtimes[key]


class DadsCloudServicer:
    def __init__(self, device: str):
        self.cache = CloudRuntimeCache(device)

    def GetCloudProfile(self, request: dict[str, Any], context) -> dict[str, Any]:
        try:
            profile = self.cache.get_profile(
                model_name=request["model_name"],
                input_shape=[int(item) for item in request.get("input_shape", [1, 3, 224, 224])],
                warmup_runs=int(request.get("warmup_runs", 1)),
                profile_runs=int(request.get("profile_runs", 1)),
            )
            return {"status": "ok", "profile": profile.to_dict(), "error_message": ""}
        except Exception as exc:
            return {"status": "error", "profile": None, "error_message": str(exc)}

    def RunPartition(self, request: dict[str, Any], context) -> dict[str, Any]:
        try:
            model_name = request["model_name"]
            cloud_nodes = [str(item) for item in request.get("cloud_nodes", [])]
            profile_node_ids = set(str(item) for item in request.get("profile_node_ids", []))
            runtime = self.cache.get_runtime(model_name, profile_node_ids)
            tensors = {
                payload["node_id"]: payload_to_tensor(payload, runtime.torch, runtime.device)
                for payload in request.get("tensors", [])
            }
            cloud_actual_ms, output = execute_cloud_partition(runtime, tensors, cloud_nodes)
            output_shape = list(output.shape) if hasattr(output, "shape") else []
            return {
                "status": "ok",
                "cloud_actual_ms": cloud_actual_ms,
                "output_shape": output_shape,
                "error_message": "",
            }
        except Exception as exc:
            return {
                "status": "error",
                "cloud_actual_ms": 0.0,
                "output_shape": [],
                "error_message": str(exc),
            }


def serve(host: str, port: int, device: str, max_workers: int = 4) -> None:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers), options=grpc_options())
    add_dads_cloud_servicer_to_server(DadsCloudServicer(device), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print(f"DADS cloud server listening on {host}:{port} with device={device}")
    server.wait_for_termination()
