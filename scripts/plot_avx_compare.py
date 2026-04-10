#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_one_image(non_avx_csv: Path, avx_csv: Path, out_dir: Path) -> None:
    df_no = pd.read_csv(non_avx_csv).sort_values("threads").reset_index(drop=True)
    df_avx = pd.read_csv(avx_csv).sort_values("threads").reset_index(drop=True)

    label = non_avx_csv.stem

    plt.figure()
    plt.plot(df_no["threads"], df_no["mean_s"], marker="o", label="without AVX2")
    plt.plot(df_avx["threads"], df_avx["mean_s"], marker="o", label="with AVX2")
    plt.title(f"Execution Time vs Threads ({label})")
    plt.xlabel("Threads")
    plt.ylabel("Time (s)")
    plt.xticks(df_no["threads"].tolist())
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_time_compare.png", dpi=150)
    plt.close()


def plot_all_on_one(non_avx_dir: Path, avx_dir: Path, out_dir: Path) -> None:
    files = sorted([p.name for p in non_avx_dir.glob("*.csv") if (avx_dir / p.name).exists()])

    plt.figure()
    for name in files:
        df_no = pd.read_csv(non_avx_dir / name).sort_values("threads")
        df_avx = pd.read_csv(avx_dir / name).sort_values("threads")
        label = Path(name).stem
        plt.plot(df_no["threads"], df_no["mean_s"], marker="o", linestyle="-", label=f"{label} no AVX2")
        plt.plot(df_avx["threads"], df_avx["mean_s"], marker="o", linestyle="--", label=f"{label} AVX2")

    plt.title("Execution Time vs Threads (AVX2 vs without AVX2)")
    plt.xlabel("Threads")
    plt.ylabel("Time (s)")
    plt.xticks([1, 2, 4, 8])
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "all_images_time_compare.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AVX2 vs non-AVX2 time comparisons.")
    parser.add_argument("--without", required=True, help="Directory with per-image CSVs without AVX2")
    parser.add_argument("--with-avx2", required=True, help="Directory with per-image CSVs with AVX2")
    parser.add_argument("--out", default="plots_avx_compare", help="Output directory")
    args = parser.parse_args()

    without_dir = Path(args.without).resolve()
    with_dir = Path(args.with_avx2).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p.name for p in without_dir.glob("*.csv") if (with_dir / p.name).exists()])
    if not files:
        raise FileNotFoundError("No matching CSV files found in the two directories.")

    for name in files:
        plot_one_image(without_dir / name, with_dir / name, out_dir)

    plot_all_on_one(without_dir, with_dir, out_dir)
    print(f"Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
