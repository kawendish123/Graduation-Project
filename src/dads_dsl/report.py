from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional


SUMMARY_COLUMNS = [
    "model",
    "bandwidth_mbps",
    "cpu_load_target",
    "cpu_load_avg",
    "split_nodes",
    "edge_node_count",
    "cloud_node_count",
    "dsl_estimated_edge_ms",
    "dsl_estimated_transfer_ms",
    "dsl_estimated_cloud_ms",
    "dsl_estimated_total_ms",
    "all_edge_estimated_ms",
    "edge_actual_ms",
    "transmission_actual_ms",
    "cloud_actual_ms",
    "total_actual_ms",
    "payload_bytes",
]


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        if len(value) <= 4:
            return ",".join(str(item) for item in value)
        return ",".join(str(item) for item in value[:4]) + f"...({len(value)})"
    return str(value)


def format_table(row: dict[str, Any]) -> str:
    values = [_format_value(row.get(column, "")) for column in SUMMARY_COLUMNS]
    widths = [max(len(column), len(value)) for column, value in zip(SUMMARY_COLUMNS, values)]
    header = " | ".join(column.ljust(width) for column, width in zip(SUMMARY_COLUMNS, widths))
    sep = "-+-".join("-" * width for width in widths)
    body = " | ".join(value.ljust(width) for value, width in zip(values, widths))
    return "\n".join([header, sep, body])


def write_report(row: dict[str, Any], full_payload: dict[str, Any], report_format: str, report_output: Optional[str]) -> str:
    if report_format == "json":
        rendered = json.dumps(full_payload, indent=2)
    elif report_format == "csv":
        rendered = ",".join(SUMMARY_COLUMNS) + "\n" + ",".join(_format_value(row.get(column, "")) for column in SUMMARY_COLUMNS)
    else:
        rendered = format_table(row)

    if report_output:
        output = Path(report_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if report_format == "csv":
            with output.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
                writer.writeheader()
                csv_row = dict(row)
                csv_row["split_nodes"] = ";".join(row.get("split_nodes", []))
                writer.writerow({column: csv_row.get(column, "") for column in SUMMARY_COLUMNS})
        else:
            output.write_text(rendered, encoding="utf-8")
    return rendered
