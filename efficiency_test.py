#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


@dataclass
class BenchmarkRow:
    threads: int
    run_index: int
    elapsed_sec: float


@dataclass
class SummaryRow:
    threads: int
    runs: int
    min_sec: float
    mean_sec: float
    median_sec: float
    std_sec: float
    speedup: float
    efficiency: float
    karp_flatt: float | None


def run_once(executable: Path, image_path: Path, threads: int) -> float:
    """
    Runs the benchmark once and returns wall-clock time in seconds.

    Assumption:
        executable CLI is:
            ./shi_tomasi <image_path> <num_threads>

    If your program currently accepts only <image_path>,
    update it to accept num_threads too.
    """
    cmd = [str(executable), str(image_path), str(threads)]

    start = time.perf_counter()
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    end = time.perf_counter()

    if completed.returncode != 0:
        print("Command failed:", " ".join(cmd), file=sys.stderr)
        print("STDOUT:\n", completed.stdout, file=sys.stderr)
        print("STDERR:\n", completed.stderr, file=sys.stderr)
        raise RuntimeError(f"Benchmark run failed for threads={threads}")

    return end - start


def karp_flatt_metric(speedup: float, p: int) -> float | None:
    """
    Karp-Flatt serial fraction:
        e = (1/S(p) - 1/p) / (1 - 1/p)
    Undefined for p == 1.
    """
    if p <= 1:
        return None
    denominator = 1.0 - 1.0 / p
    if denominator == 0.0:
        return None
    return (1.0 / speedup - 1.0 / p) / denominator


def summarize(rows: List[BenchmarkRow]) -> List[SummaryRow]:
    grouped: dict[int, List[float]] = {}
    for row in rows:
        grouped.setdefault(row.threads, []).append(row.elapsed_sec)

    if 1 not in grouped:
        raise ValueError("Thread count 1 must be present to compute speedup and efficiency.")

    baseline = statistics.mean(grouped[1])

    summary: List[SummaryRow] = []
    for threads in sorted(grouped.keys()):
        values = grouped[threads]
        mean_sec = statistics.mean(values)
        std_sec = statistics.stdev(values) if len(values) > 1 else 0.0
        speedup = baseline / mean_sec
        efficiency = speedup / threads
        karp = karp_flatt_metric(speedup, threads)

        summary.append(
            SummaryRow(
                threads=threads,
                runs=len(values),
                min_sec=min(values),
                mean_sec=mean_sec,
                median_sec=statistics.median(values),
                std_sec=std_sec,
                speedup=speedup,
                efficiency=efficiency,
                karp_flatt=karp,
            )
        )

    return summary


def save_raw_csv(rows: List[BenchmarkRow], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threads", "run_index", "elapsed_sec"])
        for row in rows:
            writer.writerow([row.threads, row.run_index, f"{row.elapsed_sec:.9f}"])


def save_summary_csv(summary: List[SummaryRow], out_path: Path) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "threads",
                "runs",
                "min_sec",
                "mean_sec",
                "median_sec",
                "std_sec",
                "speedup",
                "efficiency",
                "karp_flatt",
            ]
        )
        for row in summary:
            writer.writerow(
                [
                    row.threads,
                    row.runs,
                    f"{row.min_sec:.9f}",
                    f"{row.mean_sec:.9f}",
                    f"{row.median_sec:.9f}",
                    f"{row.std_sec:.9f}",
                    f"{row.speedup:.9f}",
                    f"{row.efficiency:.9f}",
                    "" if row.karp_flatt is None else f"{row.karp_flatt:.9f}",
                ]
            )


def plot_time(summary: List[SummaryRow], out_dir: Path) -> None:
    threads = [r.threads for r in summary]
    means = [r.mean_sec for r in summary]
    stds = [r.std_sec for r in summary]

    plt.figure(figsize=(8, 5))
    plt.errorbar(threads, means, yerr=stds, marker="o", capsize=4)
    plt.xlabel("Threads")
    plt.ylabel("Execution time, s")
    plt.title("Execution time vs number of threads")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "time_vs_threads.png", dpi=160)
    plt.close()


def plot_speedup(summary: List[SummaryRow], out_dir: Path) -> None:
    threads = [r.threads for r in summary]
    speedups = [r.speedup for r in summary]
    ideal = threads

    plt.figure(figsize=(8, 5))
    plt.plot(threads, speedups, marker="o", label="Measured speedup")
    plt.plot(threads, ideal, linestyle="--", label="Ideal speedup")
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.title("Speedup vs number of threads")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_vs_threads.png", dpi=160)
    plt.close()


def plot_efficiency(summary: List[SummaryRow], out_dir: Path) -> None:
    threads = [r.threads for r in summary]
    efficiencies = [r.efficiency for r in summary]

    plt.figure(figsize=(8, 5))
    plt.plot(threads, efficiencies, marker="o")
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Threads")
    plt.ylabel("Parallel efficiency")
    plt.title("Parallel efficiency vs number of threads")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "efficiency_vs_threads.png", dpi=160)
    plt.close()


def plot_karp_flatt(summary: List[SummaryRow], out_dir: Path) -> None:
    filtered = [r for r in summary if r.karp_flatt is not None]
    if not filtered:
        return

    threads = [r.threads for r in filtered]
    karp = [r.karp_flatt for r in filtered]

    plt.figure(figsize=(8, 5))
    plt.plot(threads, karp, marker="o")
    plt.xlabel("Threads")
    plt.ylabel("Karp-Flatt serial fraction")
    plt.title("Karp-Flatt metric vs number of threads")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "karp_flatt_vs_threads.png", dpi=160)
    plt.close()


def parse_thread_list(value: str) -> List[int]:
    result: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n <= 0:
            raise ValueError("Thread counts must be positive.")
        result.append(n)

    unique_sorted = sorted(set(result))
    if 1 not in unique_sorted:
        unique_sorted.insert(0, 1)
    return unique_sorted


def print_summary(summary: List[SummaryRow]) -> None:
    header = (
        f"{'thr':>4} | {'mean(s)':>10} | {'std(s)':>10} | "
        f"{'speedup':>10} | {'eff':>10} | {'karp-flatt':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        karp = "-" if row.karp_flatt is None else f"{row.karp_flatt:.6f}"
        print(
            f"{row.threads:>4} | "
            f"{row.mean_sec:>10.6f} | "
            f"{row.std_sec:>10.6f} | "
            f"{row.speedup:>10.4f} | "
            f"{row.efficiency:>10.4f} | "
            f"{karp:>12}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark parallel Shi-Tomasi implementation."
    )
    parser.add_argument(
        "--exe",
        required=True,
        help="Path to executable, e.g. ./build/shi_tomasi",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to test image",
    )
    parser.add_argument(
        "--threads",
        default="1,2,4,8",
        help="Comma-separated thread counts, e.g. 1,2,4,8,12",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of runs per thread count",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per thread count",
    )
    parser.add_argument(
        "--out",
        default="benchmark_results",
        help="Output directory",
    )

    args = parser.parse_args()

    executable = Path(args.exe).resolve()
    image_path = Path(args.image).resolve()
    out_dir = Path(args.out).resolve()

    if not executable.exists():
        raise FileNotFoundError(f"Executable not found: {executable}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")

    thread_counts = parse_thread_list(args.threads)

    out_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: List[BenchmarkRow] = []

    print("Benchmark configuration:")
    print(f"  Executable: {executable}")
    print(f"  Image:      {image_path}")
    print(f"  Threads:    {thread_counts}")
    print(f"  Repeats:    {args.repeats}")
    print(f"  Warmup:     {args.warmup}")
    print(f"  Output dir: {out_dir}")
    print()

    for threads in thread_counts:
        print(f"[threads={threads}] warmup...", flush=True)
        for _ in range(args.warmup):
            run_once(executable, image_path, threads)

        print(f"[threads={threads}] measuring...", flush=True)
        for run_idx in range(args.repeats):
            elapsed = run_once(executable, image_path, threads)
            raw_rows.append(BenchmarkRow(threads=threads, run_index=run_idx, elapsed_sec=elapsed))
            print(f"  run {run_idx + 1}/{args.repeats}: {elapsed:.6f} s", flush=True)

    summary = summarize(raw_rows)

    save_raw_csv(raw_rows, out_dir / "raw_results.csv")
    save_summary_csv(summary, out_dir / "summary.csv")

    plot_time(summary, out_dir)
    plot_speedup(summary, out_dir)
    plot_efficiency(summary, out_dir)
    plot_karp_flatt(summary, out_dir)

    print()
    print_summary(summary)
    print()
    print(f"Saved results to: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
