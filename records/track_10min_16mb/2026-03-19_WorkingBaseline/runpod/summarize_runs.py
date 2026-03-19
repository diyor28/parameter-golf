#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return "-"
    return str(value)


def row(summary: dict) -> dict[str, object]:
    final_eval = summary.get("final_int8_zlib_roundtrip") or {}
    last_val = summary.get("last_val") or {}
    last_train = summary.get("last_train") or {}
    return {
        "run_id": summary.get("run_id"),
        "status": summary.get("status", "unknown"),
        "steps": summary.get("completed_steps"),
        "best_val_bpb": summary.get("best_val_bpb"),
        "final_bpb": final_eval.get("val_bpb"),
        "final_val_loss": final_eval.get("val_loss"),
        "last_train_loss": last_train.get("train_loss"),
        "train_shards": summary.get("actual_train_shards"),
        "train_time_ms": summary.get("training_time_ms"),
        "quantized_bytes": summary.get("quantized_model_bytes"),
        "submission_bytes": summary.get("quantized_submission_size_bytes"),
        "peak_mem_mib": summary.get("peak_memory_allocated_mib"),
        "path": summary.get("experiment_dir"),
        "last_val_step": last_val.get("step"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize record runs from logs/record_runs")
    parser.add_argument("runs_dir", nargs="?", default="logs/record_runs")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise SystemExit(f"Runs directory not found: {runs_dir}")

    summaries = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        summary = load_summary(summary_path)
        if summary is not None:
            summaries.append(row(summary))

    summaries.sort(key=lambda item: (item["final_bpb"] is None, item["final_bpb"]))

    if not summaries:
        raise SystemExit(f"No summary.json files found under {runs_dir}")

    header = [
        "run_id",
        "status",
        "steps",
        "best_val_bpb",
        "final_bpb",
        "final_val_loss",
        "last_train_loss",
        "train_shards",
        "train_time_ms",
        "submission_bytes",
        "peak_mem_mib",
    ]
    print("\t".join(header))
    for item in summaries:
        print(
            "\t".join(
                [
                    fmt(item["run_id"]),
                    fmt(item["status"]),
                    fmt(item["steps"]),
                    fmt(item["best_val_bpb"]),
                    fmt(item["final_bpb"]),
                    fmt(item["final_val_loss"]),
                    fmt(item["last_train_loss"]),
                    fmt(item["train_shards"]),
                    fmt(item["train_time_ms"], digits=0),
                    fmt(item["submission_bytes"], digits=0),
                    fmt(item["peak_mem_mib"], digits=0),
                ]
            )
        )


if __name__ == "__main__":
    main()
