from __future__ import annotations

from typing import Any

import numpy as np


_TORCH_TO_NUMPY = {
    "torch.float32": np.float32,
    "torch.float64": np.float64,
    "torch.float16": np.float16,
    "torch.int64": np.int64,
    "torch.int32": np.int32,
    "torch.int16": np.int16,
    "torch.int8": np.int8,
    "torch.uint8": np.uint8,
    "torch.bool": np.bool_,
}


def tensor_to_payload(node_id: str, tensor: Any) -> dict[str, Any]:
    tensor = tensor.detach().cpu().contiguous()
    dtype = str(tensor.dtype)
    if dtype not in _TORCH_TO_NUMPY:
        raise ValueError(f"Unsupported tensor dtype for serialization: {dtype}")
    return {
        "node_id": node_id,
        "dtype": dtype,
        "shape": list(tensor.shape),
        "raw_bytes": tensor.numpy().tobytes(),
    }


def payload_to_tensor(payload: dict[str, Any], torch_mod: Any, device: Any = None) -> Any:
    dtype = str(payload["dtype"])
    if dtype not in _TORCH_TO_NUMPY:
        raise ValueError(f"Unsupported tensor dtype for deserialization: {dtype}")
    array = np.frombuffer(payload["raw_bytes"], dtype=_TORCH_TO_NUMPY[dtype]).copy()
    array = array.reshape(tuple(int(item) for item in payload["shape"]))
    tensor = torch_mod.from_numpy(array)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def payload_nbytes(payload: dict[str, Any]) -> int:
    return len(payload["raw_bytes"])
