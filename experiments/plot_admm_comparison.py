"""Visualize ADMM convergence comparison between CPU and GPU."""

import argparse
import json
import os

import matplotlib.pyplot as plt

# Enable LaTeX rendering
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.labelsize": 14,
        "font.size": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }
)


def load_results(dataset_name: str, device: str, results_dir: str, seed: int = 42):
    """Load ADMM iteration results from JSON file.

    Args:
        dataset_name: Name of the dataset
        device: Device name ('cpu', 'cuda', etc.)
        results_dir: Directory containing result files
        seed: Random seed used in experiment

    Returns:
        dict: Loaded results dictionary
    """
    json_filename = (
        f"admm_iterations_{dataset_name}_{device.replace(':', '_')}_seed{seed}.json"
    )
    json_path = os.path.join(results_dir, json_filename)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Results file not found: {json_path}")

    with open(json_path, "r") as f:
        results = json.load(f)

    return results


def plot_residuals_vs_time(
    cpu_results: dict,
    gpu_results: dict,
    output_path: str = None,
    figsize: tuple = (10, 10),
):
    """Plot primal and dual residual norms vs time for CPU and GPU.

    Args:
        cpu_results: Results dictionary from CPU run
        gpu_results: Results dictionary from GPU run
        output_path: Path to save the plot (if None, displays instead)
        figsize: Figure size (width, height)
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # Plot primal residual
    ax1.plot(
        cpu_results["cumulative_times"],
        cpu_results["primal_residual_norms"],
        label=r"CPU",
        linewidth=2,
        marker="o",
        markersize=3,
        markevery=max(1, len(cpu_results["cumulative_times"]) // 20),
    )
    ax1.plot(
        gpu_results["cumulative_times"],
        gpu_results["primal_residual_norms"],
        label=r"GPU",
        linewidth=2,
        marker="s",
        markersize=3,
        markevery=max(1, len(gpu_results["cumulative_times"]) // 20),
    )
    ax1.set_xlabel(r"Time (seconds)")
    ax1.set_ylabel(r"Primal Residual Norm")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot dual residual
    ax2.plot(
        cpu_results["cumulative_times"],
        cpu_results["dual_residual_norms"],
        label=r"CPU",
        linewidth=2,
        marker="o",
        markersize=3,
        markevery=max(1, len(cpu_results["cumulative_times"]) // 20),
    )
    ax2.plot(
        gpu_results["cumulative_times"],
        gpu_results["dual_residual_norms"],
        label=r"GPU",
        linewidth=2,
        marker="s",
        markersize=3,
        markevery=max(1, len(gpu_results["cumulative_times"]) // 20),
    )
    ax2.set_xlabel(r"Time (seconds)")
    ax2.set_ylabel(r"Dual Residual Norm")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle(
        rf"ADMM Convergence: {cpu_results['dataset']} (CPU vs GPU)",
        fontsize=16,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Main function to create comparison plots."""
    parser = argparse.ArgumentParser(
        description="Plot ADMM convergence comparison between CPU and GPU"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["acsincome", "e2006", "realsim", "yearpredictionmsd"],
        help="Dataset name",
    )
    parser.add_argument(
        "--cpu-device",
        type=str,
        default="cpu",
        help="CPU device name used in results filename",
    )
    parser.add_argument(
        "--gpu-device",
        type=str,
        default="cuda",
        help="GPU device name used in results filename",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./experiment_results",
        help="Directory containing result files",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed used in experiments"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (if not specified, displays plot)",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[10, 10],
        help="Figure size (width height)",
    )

    args = parser.parse_args()

    try:
        # Load results
        print(f"Loading CPU results for {args.dataset}...")
        cpu_results = load_results(
            args.dataset, args.cpu_device, args.results_dir, args.seed
        )

        print(f"Loading GPU results for {args.dataset}...")
        gpu_results = load_results(
            args.dataset, args.gpu_device, args.results_dir, args.seed
        )

        # Generate output path if not specified
        output_path = args.output
        if output_path is None:
            output_path = os.path.join(
                args.results_dir, f"admm_comparison_{args.dataset}_seed{args.seed}.pdf"
            )

        # Create plot
        print("Creating comparison plot...")
        plot_residuals_vs_time(
            cpu_results, gpu_results, output_path, figsize=tuple(args.figsize)
        )

        print("\nSummary:")
        print(f"Dataset: {args.dataset}")
        print(f"CPU total time: {cpu_results['total_solve_time']:.2f}s")
        print(f"GPU total time: {gpu_results['total_solve_time']:.2f}s")
        print(
            f"Speedup: {cpu_results['total_solve_time'] / gpu_results['total_solve_time']:.2f}x"
        )
        print(f"CPU final primal residual: {cpu_results['final_primal_residual']:.4e}")
        print(f"GPU final primal residual: {gpu_results['final_primal_residual']:.4e}")
        print(f"CPU final dual residual: {cpu_results['final_dual_residual']:.4e}")
        print(f"GPU final dual residual: {gpu_results['final_dual_residual']:.4e}")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(
            "\nMake sure you have run admm_iteration_timing.py for both CPU and GPU first."
        )
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
