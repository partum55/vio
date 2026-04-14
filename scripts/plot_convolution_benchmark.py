import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python plot_convolution_benchmark.py <csv_path> [output_prefix]")
        return 1

    csv_path = Path(sys.argv[1])
    output_prefix = Path(sys.argv[2]) if len(sys.argv) >= 3 else csv_path.with_suffix("")

    df = pd.read_csv(csv_path).sort_values("threads")

    time_path = f"{output_prefix}_time.png"
    speedup_path = f"{output_prefix}_speedup.png"
    efficiency_path = f"{output_prefix}_efficiency.png"

    plt.figure(figsize=(8, 5))
    plt.plot(df["threads"], df["mean_ms"], marker="o")
    plt.xlabel("Threads")
    plt.ylabel("Mean time (ms)")
    plt.title("Convolution benchmark: execution time")
    plt.grid(True)
    plt.xticks(df["threads"])
    plt.tight_layout()
    plt.savefig(time_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["threads"], df["speedup"], marker="o", label="Measured speedup")
    plt.plot(df["threads"], df["threads"], linestyle="--", label="Ideal speedup")
    plt.xlabel("Threads")
    plt.ylabel("Speedup = T1 / Tp")
    plt.title("Convolution benchmark: speedup")
    plt.grid(True)
    plt.xticks(df["threads"])
    plt.legend()
    plt.tight_layout()
    plt.savefig(speedup_path, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["threads"], df["efficiency"], marker="o")
    plt.xlabel("Threads")
    plt.ylabel("Efficiency = Speedup / p")
    plt.title("Convolution benchmark: efficiency")
    plt.grid(True)
    plt.xticks(df["threads"])
    plt.tight_layout()
    plt.savefig(efficiency_path, dpi=150)
    plt.close()

    print(f"Saved: {time_path}")
    print(f"Saved: {speedup_path}")
    print(f"Saved: {efficiency_path}")

    best_row = df.loc[df["mean_ms"].idxmin()]
    print("\nBest result:")
    print(best_row.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
