from __future__ import annotations

import pickle
from typing import Any, Callable

import grpc

SERVICE_NAME = "dads_dsl.DadsCloud"
GET_CLOUD_PROFILE = f"/{SERVICE_NAME}/GetCloudProfile"
RUN_PARTITION = f"/{SERVICE_NAME}/RunPartition"

_MAX_MESSAGE_BYTES = 512 * 1024 * 1024


def grpc_options() -> list[tuple[str, int]]:
    return [
        ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
    ]


def serialize_message(message: dict[str, Any]) -> bytes:
    return pickle.dumps(message, protocol=4)


def deserialize_message(payload: bytes) -> dict[str, Any]:
    return pickle.loads(payload)


class DadsCloudStub:
    def __init__(self, channel: grpc.Channel):
        self.get_cloud_profile = channel.unary_unary(
            GET_CLOUD_PROFILE,
            request_serializer=serialize_message,
            response_deserializer=deserialize_message,
        )
        self.run_partition = channel.unary_unary(
            RUN_PARTITION,
            request_serializer=serialize_message,
            response_deserializer=deserialize_message,
        )


def add_dads_cloud_servicer_to_server(servicer: Any, server: grpc.Server) -> None:
    handlers: dict[str, grpc.RpcMethodHandler] = {
        "GetCloudProfile": grpc.unary_unary_rpc_method_handler(
            servicer.GetCloudProfile,
            request_deserializer=deserialize_message,
            response_serializer=serialize_message,
        ),
        "RunPartition": grpc.unary_unary_rpc_method_handler(
            servicer.RunPartition,
            request_deserializer=deserialize_message,
            response_serializer=serialize_message,
        ),
    }
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(SERVICE_NAME, handlers),))


def make_channel(target: str) -> grpc.Channel:
    return grpc.insecure_channel(target, options=grpc_options())
