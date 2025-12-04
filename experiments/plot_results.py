"""Plot experiment results comparing GPU vs CPU solve times."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Use LaTeX for text rendering
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    }
)


def load_all_results(results_dir: str = "./experiment_results") -> pd.DataFrame:
    """Load all CSV files from the results directory.

    Args:
        results_dir: Directory containing result CSV files

    Returns:
        Combined DataFrame with all results
    """
    results_path = Path(results_dir)
    csv_files = list(results_path.glob("*.csv"))

    if not csv_files:
        raise ValueError(f"No CSV files found in {results_dir}")

    print(f"Found {len(csv_files)} result files")

    # Load and combine all CSVs
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} total results")

    return combined_df


def create_comparison_plot(
    df: pd.DataFrame,
    output_path: str = "./experiment_results/solve_time_comparison.pdf",
):
    """Create bar chart comparing GPU vs CPU solve times.

    Args:
        df: DataFrame with experiment results
        output_path: Path to save the PDF plot
    """
    # Filter for converged results only
    converged_df = df[df["solver_status"] == "ConvergenceStatus.CONVERGED"].copy()

    print(f"\nConverged results: {len(converged_df)}/{len(df)}")

    if len(converged_df) == 0:
        print("WARNING: No converged results found!")
        return

    # Show which experiments didn't converge
    non_converged = df[df["solver_status"] != "ConvergenceStatus.CONVERGED"]
    if len(non_converged) > 0:
        print("\nNon-converged experiments:")
        for _, row in non_converged.iterrows():
            print(f"  - {row['dataset']} on {row['device']}: {row['solver_status']}")

    # Normalize device names
    converged_df["device_type"] = converged_df["device"].apply(
        lambda x: "GPU" if x.startswith("cuda") else "CPU"
    )

    # Group by dataset and device type
    grouped = (
        converged_df.groupby(["dataset", "device_type"])["total_time"]
        .mean()
        .reset_index()
    )

    # Pivot to get CPU and GPU columns
    pivot_df = grouped.pivot(
        index="dataset", columns="device_type", values="total_time"
    )

    # Get dataset dimensions for labels
    dataset_info = (
        converged_df.groupby("dataset")
        .agg({"n_samples": "first", "n_features": "first"})
        .reset_index()
    )

    # Create labels with dataset name and dimensions
    labels = []
    for _, row in dataset_info.iterrows():
        dataset = row["dataset"]
        n_samples = int(row["n_samples"])
        n_features = int(row["n_features"])
        labels.append(f"{dataset}\n({n_samples} × {n_features})")

    # Sort by dataset name to ensure consistent ordering
    dataset_info = dataset_info.sort_values("dataset")
    pivot_df = pivot_df.reindex(dataset_info["dataset"])

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(pivot_df))
    width = 0.35

    # Plot bars
    cpu_times = pivot_df.get("CPU", pd.Series([None] * len(pivot_df)))
    gpu_times = pivot_df.get("GPU", pd.Series([None] * len(pivot_df)))

    bars1 = ax.bar([i - width / 2 for i in x], cpu_times, width, label="CPU", alpha=0.8)
    bars2 = ax.bar([i + width / 2 for i in x], gpu_times, width, label="GPU", alpha=0.8)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height is not None and not pd.isna(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}s",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    add_value_labels(bars1)
    add_value_labels(bars2)

    # Customize plot
    ax.set_xlabel(r"Dataset (samples $\times$ features)", fontsize=12)
    ax.set_ylabel(r"Solve Time (seconds)", fontsize=12)
    ax.set_title(
        r"\textbf{ADMM Solver: GPU vs CPU Performance Comparison}", fontsize=14
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Set y-axis to log scale
    ax.set_yscale("log")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=300)
    print(f"\nPlot saved to: {output_path}")

    # Also show summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics (Converged Results Only)")
    print("=" * 60)
    print(pivot_df.to_string())

    # Calculate speedup
    if "GPU" in pivot_df.columns and "CPU" in pivot_df.columns:
        speedup = pivot_df["CPU"] / pivot_df["GPU"]
        print("\n" + "=" * 60)
        print("GPU Speedup (CPU time / GPU time)")
        print("=" * 60)
        for dataset, speed in speedup.items():
            if not pd.isna(speed):
                print(f"{dataset:20s}: {speed:.2f}x")


def main():
    """Main function to load results and create plot."""
    results_dir = "./experiment_results"

    # Check if results directory exists
    if not os.path.exists(results_dir):
        print(f"ERROR: Results directory '{results_dir}' not found!")
        print("Please run experiments first using ./run_all_experiments.sh")
        return

    # Load results
    df = load_all_results(results_dir)

    # Create comparison plot
    create_comparison_plot(df)


if __name__ == "__main__":
    main()
