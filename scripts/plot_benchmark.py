#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_plot_single(x, y, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel("Threads")
    plt.ylabel(ylabel)
    plt.xticks(list(x))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_plot_multi(dfs, labels, column: str, title: str, ylabel: str, out_path: Path) -> None:
    plt.figure()
    for df, label in zip(dfs, labels):
        plt.plot(df["threads"], df[column], marker="o", label=label)

    plt.title(title)
    plt.xlabel("Threads")
    plt.ylabel(ylabel)
    plt.xticks(sorted(dfs[0]["threads"].tolist()))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def amdahl_speedup(serial_fraction: float, threads: int) -> float:
    return 1.0 / (serial_fraction + (1.0 - serial_fraction) / threads)


def save_speedup_single(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure()
    plt.plot(df["threads"], df["speedup"], marker="o", label="Measured speedup")

    if "amdahl_serial_fraction" in df.columns:
        f = float(df.loc[df["threads"] == df["threads"].max(), "amdahl_serial_fraction"].iloc[0])
        predicted = [amdahl_speedup(f, int(p)) for p in df["threads"]]
        plt.plot(df["threads"], predicted, marker="o", linestyle="--", label=f"Amdahl fit (f={f:.3f})")

    plt.title("Speedup vs Threads")
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.xticks(df["threads"].tolist())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_speedup_multi(dfs, labels, out_path: Path) -> None:
    plt.figure()
    for df, label in zip(dfs, labels):
        plt.plot(df["threads"], df["speedup"], marker="o", label=label)

    plt.title("Speedup vs Threads")
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.xticks(sorted(dfs[0]["threads"].tolist()))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_amdahl_fit_single(df: pd.DataFrame, out_path: Path) -> None:
    if "amdahl_serial_fraction" not in df.columns:
        return

    f = float(df.loc[df["threads"] == df["threads"].max(), "amdahl_serial_fraction"].iloc[0])
    predicted = [amdahl_speedup(f, int(p)) for p in df["threads"]]

    plt.figure()
    plt.plot(df["threads"], df["speedup"], marker="o", label="Measured")
    plt.plot(df["threads"], predicted, marker="o", linestyle="--", label="Amdahl prediction")
    plt.title(f"Amdahl Fit vs Measured Speedup (f={f:.3f})")
    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.xticks(df["threads"].tolist())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark plots from one or more CSV files.")
    parser.add_argument("--csv", nargs="+", required=True, help="One or more CSV files")
    parser.add_argument("--out", default="plots", help="Output directory for plots")
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(p).resolve() for p in args.csv]

    for csv_path in csv_paths:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    dfs = []
    labels = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path).sort_values("threads").reset_index(drop=True)

        required = {"threads", "speedup", "efficiency", "mean_s", "time_reduction"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path.name} is missing required columns: {sorted(missing)}")

        dfs.append(df)
        labels.append(csv_path.stem)

    if len(dfs) == 1:
        df = dfs[0]

        save_speedup_single(df, out_dir / "speedup.png")
        save_plot_single(df["threads"], df["efficiency"], "Efficiency vs Threads", "Efficiency", out_dir / "efficiency.png")
        save_plot_single(df["threads"], df["mean_s"], "Execution Time vs Threads", "Time (s)", out_dir / "time.png")
        save_plot_single(df["threads"], df["time_reduction"], "Time Reduction vs Threads", "Time Reduction", out_dir / "time_reduction.png")

        if "karp_flatt" in df.columns:
            kf = df[df["threads"] > 1]
            if not kf.empty:
                save_plot_single(kf["threads"], kf["karp_flatt"], "Karp-Flatt Metric vs Threads", "Karp-Flatt", out_dir / "karp_flatt.png")

        if "scaling_quality" in df.columns:
            save_plot_single(df["threads"], df["scaling_quality"], "Scaling Quality vs Threads", "Scaling Quality", out_dir / "scaling_quality.png")

        save_amdahl_fit_single(df, out_dir / "amdahl_fit.png")

    else:
        save_speedup_multi(dfs, labels, out_dir / "speedup.png")
        save_plot_multi(dfs, labels, "efficiency", "Efficiency vs Threads", "Efficiency", out_dir / "efficiency.png")
        save_plot_multi(dfs, labels, "mean_s", "Execution Time vs Threads", "Time (s)", out_dir / "time.png")
        save_plot_multi(dfs, labels, "time_reduction", "Time Reduction vs Threads", "Time Reduction", out_dir / "time_reduction.png")

        if all("karp_flatt" in df.columns for df in dfs):
            dfs_kf = [df[df["threads"] > 1] for df in dfs]
            save_plot_multi(dfs_kf, labels, "karp_flatt", "Karp-Flatt Metric vs Threads", "Karp-Flatt", out_dir / "karp_flatt.png")

        if all("scaling_quality" in df.columns for df in dfs):
            save_plot_multi(dfs, labels, "scaling_quality", "Scaling Quality vs Threads", "Scaling Quality", out_dir / "scaling_quality.png")

    print(f"Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
