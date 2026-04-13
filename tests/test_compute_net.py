from dads_dsl.dsl import compute_transmission_ms


def test_compute_transmission_ms_uses_mbps_and_returns_ms():
    transmission_ms = compute_transmission_ms(output_bytes=1_250_000, bandwidth_mbps=10.0)
    assert transmission_ms == 1000.0


def test_compute_transmission_ms_rejects_non_positive_bandwidth():
    try:
        compute_transmission_ms(output_bytes=128, bandwidth_mbps=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive bandwidth.")
