from dads_dsl.report import format_table


def test_format_table_contains_latency_columns():
    table = format_table(
        {
            "model": "mobilenet_v2",
            "bandwidth_mbps": 5.0,
            "cpu_load_target": 0.0,
            "cpu_load_avg": 1.0,
            "split_nodes": ["__input__"],
            "edge_node_count": 1,
            "cloud_node_count": 153,
            "dsl_estimated_transfer_ms": 100.0,
            "edge_actual_ms": 1.0,
            "transmission_actual_ms": 2.0,
            "cloud_actual_ms": 3.0,
            "total_actual_ms": 6.0,
            "payload_bytes": 1024,
        }
    )
    assert "split_nodes" in table
    assert "edge_actual_ms" in table
    assert "transmission_actual_ms" in table
    assert "cloud_actual_ms" in table
    assert "total_actual_ms" in table
