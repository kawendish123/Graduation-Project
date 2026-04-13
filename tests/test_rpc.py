from concurrent import futures

import pytest

grpc = pytest.importorskip("grpc")

from dads_dsl.rpc import DadsCloudStub, add_dads_cloud_servicer_to_server, grpc_options


class _FakeServicer:
    def GetCloudProfile(self, request, context):
        return {"status": "ok", "profile": {"model_name": request["model_name"]}, "error_message": ""}

    def RunPartition(self, request, context):
        return {"status": "ok", "cloud_actual_ms": 1.5, "output_shape": [1, 1000], "error_message": ""}


def test_generic_grpc_transport_roundtrip():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1), options=grpc_options())
    add_dads_cloud_servicer_to_server(_FakeServicer(), server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    try:
        stub = DadsCloudStub(grpc.insecure_channel(f"localhost:{port}", options=grpc_options()))
        profile = stub.get_cloud_profile({"model_name": "demo"})
        result = stub.run_partition({"model_name": "demo"})
        assert profile["profile"]["model_name"] == "demo"
        assert result["cloud_actual_ms"] == 1.5
    finally:
        server.stop(0)
