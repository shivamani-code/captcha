from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_HEADER = [
    "avg_mouse_speed",
    "mouse_path_entropy",
    "click_delay",
    "task_completion_time",
    "idle_time",
    "micro_jitter_variance",
    "acceleration_curve",
    "curvature_variance",
    "overshoot_correction_ratio",
    "timing_entropy",
    "label",
]


def read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV is empty: {path}")

    header = rows[0]
    data = rows[1:]
    return header, data


def validate_header(header: list[str], path: Path) -> None:
    if header != REQUIRED_HEADER:
        raise ValueError(
            "Header mismatch.\n"
            f"File: {path}\n"
            f"Expected: {REQUIRED_HEADER}\n"
            f"Found:    {header}\n"
        )


def append_rows(dataset_path: Path, row_files: list[Path]) -> int:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    ds_header, _ = read_csv(dataset_path)
    validate_header(ds_header, dataset_path)

    appended = 0
    with dataset_path.open("a", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        for rf in row_files:
            header, rows = read_csv(rf)
            validate_header(header, rf)

            for row in rows:
                if not row:
                    continue
                if len(row) != len(REQUIRED_HEADER):
                    raise ValueError(
                        f"Row length mismatch in {rf}: expected {len(REQUIRED_HEADER)} got {len(row)}"
                    )
                writer.writerow(row)
                appended += 1

    return appended


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = repo_root / "dataset" / "behavior_data.csv"

    import argparse

    parser = argparse.ArgumentParser(
        description="Append exported SmartCAPTCHA row CSV files into dataset/behavior_data.csv"
    )
    parser.add_argument(
        "rows",
        nargs="+",
        help="One or more exported row CSV files (each contains header + single row)",
    )
    args = parser.parse_args()

    row_files = [Path(p).resolve() for p in args.rows]
    count = append_rows(dataset_path, row_files)
    print(f"Appended {count} rows into {dataset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
