#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd


LINE_RE = re.compile(
    r"threads=(?P<threads>\d+)\s+"
    r"detector_tasks=(?P<detector_tasks>\d+)\s+"
    r"repeats=(?P<repeats>\d+)\s+"
    r"warmup=(?P<warmup>\d+)\s+"
    r"mean_ms=(?P<mean_ms>[0-9]*\.?[0-9]+)\s+"
    r"std_ms=(?P<std_ms>[0-9]*\.?[0-9]+)\s+"
    r"avg_points=(?P<avg_points>[0-9]*\.?[0-9]+)"
)


def parse_output(text: str) -> Dict[str, float]:
    match = LINE_RE.search(text)
    if not match:
        raise ValueError(f"Could not parse benchmark output:\n{text}")

    return {
        "threads": int(match.group("threads")),
        "detector_tasks": int(match.group("detector_tasks")),
        "repeats": int(match.group("repeats")),
        "warmup": int(match.group("warmup")),
        "mean_ms": float(match.group("mean_ms")),
        "std_ms": float(match.group("std_ms")),
        "avg_points": float(match.group("avg_points")),
    }


def karp_flatt(speedup: float, threads: int) -> float:
    if threads <= 1 or speedup <= 0:
        return 0.0
    return ((1.0 / speedup) - (1.0 / threads)) / (1.0 - (1.0 / threads))


def amdahl_speedup(serial_fraction: float, threads: int) -> float:
    return 1.0 / (serial_fraction + (1.0 - serial_fraction) / threads)


def run_one(exe: Path, image: Path, threads: int, repeats: int, warmup: int) -> Dict[str, float]:
    cmd = [str(exe), str(image), str(threads), str(repeats), str(warmup)]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = parse_output(completed.stdout)
    data["image"] = image.name
    data["image_path"] = str(image.resolve())
    data["size_bytes"] = image.stat().st_size
    return data


def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("threads").reset_index(drop=True).copy()

    if 1 not in df["threads"].tolist():
        raise ValueError("Thread list must contain 1 to compute speedup.")

    base_time_ms = float(df.loc[df["threads"] == 1, "mean_ms"].iloc[0])

    df["mean_s"] = df["mean_ms"] / 1000.0
    df["std_s"] = df["std_ms"] / 1000.0
    df["speedup"] = base_time_ms / df["mean_ms"]
    df["efficiency"] = df["speedup"] / df["threads"]
    df["time_reduction"] = 1.0 - (df["mean_ms"] / base_time_ms)
    df["scaling_quality"] = df.apply(
        lambda row: 1.0 if int(row["threads"]) == 1
        else (row["speedup"] - 1.0) / (row["threads"] - 1.0),
        axis=1,
    )
    df["karp_flatt"] = df.apply(
        lambda row: 0.0 if int(row["threads"]) == 1
        else karp_flatt(float(row["speedup"]), int(row["threads"])),
        axis=1,
    )
    df["amdahl_serial_fraction"] = df["karp_flatt"]

    fitted_f = float(df.loc[df["threads"] == df["threads"].max(), "amdahl_serial_fraction"].iloc[0])
    df["amdahl_speedup_pred"] = df["threads"].apply(lambda p: amdahl_speedup(fitted_f, int(p)))

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark_shi_tomasi and save results to CSV.")
    parser.add_argument("--exe", required=True, help="Path to benchmark_shi_tomasi executable")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--threads", default="1,2,4,8", help="Comma-separated thread counts")
    parser.add_argument("--repeats", type=int, default=30, help="Measured runs per thread count")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs per thread count")
    parser.add_argument("--out", help="Output CSV path; default is <image_stem>.csv in current directory")
    args = parser.parse_args()

    exe = Path(args.exe).resolve()
    image = Path(args.image).resolve()

    if not exe.exists():
        raise FileNotFoundError(f"Executable not found: {exe}")
    if not image.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    threads: List[int] = [int(x.strip()) for x in args.threads.split(",") if x.strip()]
    if 1 not in threads:
        raise ValueError("Please include thread count 1.")

    rows = []
    print(f"Benchmarking: {image.name}")
    for p in threads:
        result = run_one(exe, image, p, args.repeats, args.warmup)
        rows.append(result)
        print(
            f"threads={p:<2d} "
            f"mean_ms={result['mean_ms']:.3f} "
            f"std_ms={result['std_ms']:.3f} "
            f"avg_points={result['avg_points']:.1f}"
        )

    df = pd.DataFrame(rows)
    df = build_metrics(df)

    out_path = Path(args.out) if args.out else Path(f"{image.stem}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved CSV: {out_path.resolve()}")


if __name__ == "__main__":
    main()
