import pytest

torch = pytest.importorskip("torch")

from dads_dsl.tensor_codec import payload_to_tensor, tensor_to_payload


def test_tensor_payload_roundtrip():
    tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    payload = tensor_to_payload("x", tensor)
    restored = payload_to_tensor(payload, torch)
    assert payload["node_id"] == "x"
    assert payload["shape"] == [2, 3]
    assert torch.equal(restored, tensor)
