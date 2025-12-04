"""Run PCG solver experiments on ridge regression over multiple datasets and devices."""

import argparse
import os
import time

import pandas as pd
import torch
from linops import LinearOperator
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler

from rlaopt.linalg import LinSys, NystromConfig
from rlaopt.solvers import PCG, PCGConfig, PCGStoppingCriteria


def standardize_and_convert_to_torch(X, y):
    """Standardize features and labels and convert to PyTorch tensors."""
    X_std = StandardScaler().fit_transform(X)
    y_std = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()

    X_torch = torch.tensor(X_std)
    y_torch = torch.tensor(y_std)

    return X_torch, y_torch


def load_data(file_path: str):
    """Load data from SVM light format file."""
    X, y = load_svmlight_file(file_path)
    X_dense = X.toarray()

    X_torch, y_torch = standardize_and_convert_to_torch(X_dense, y)
    return X_torch, y_torch


def gaussian_rand_features(X, n_features, bandwidth):
    """Generate Gaussian random features."""
    W = (
        1
        / bandwidth
        * torch.randn((X.shape[1], n_features), device=X.device)
        / (n_features**0.5)
    )
    b = 2 * torch.pi * torch.rand((n_features,), device=X.device)
    return torch.cos(X @ W + b) * (2 / n_features) ** 0.5


def relu_rand_features(X, n_features):
    """Generate ReLU random features."""
    W = torch.randn((X.shape[1], n_features), device=X.device) / (n_features**0.5)
    return torch.relu(X @ W)


def load_acsincome():
    """Load ACS Income dataset."""
    X = pd.read_pickle("./data/acsincome_data.pkl")
    y = pd.read_pickle("./data/acsincome_target.pkl")
    X_torch, y_torch = standardize_and_convert_to_torch(X.to_numpy(), y.to_numpy())
    X_torch = gaussian_rand_features(X_torch, 3000, bandwidth=1.0)
    return X_torch, y_torch


def load_e2006():
    """Load E2006 dataset."""
    X_torch, y_torch = load_data("./data/E2006.train.bz2")
    return X_torch, y_torch


def load_realsim():
    """Load Real-Sim dataset."""
    X_torch, y_torch = load_data("./data/real-sim.bz2")
    return X_torch, y_torch


def load_yearpredictionmsd():
    """Load Year Prediction MSD dataset."""
    X_torch, y_torch = load_data("./data/YearPredictionMSD.bz2")
    X_torch = relu_rand_features(X_torch, X_torch.shape[0] // 100)
    return X_torch, y_torch


class GramOperator(LinearOperator):
    """Linear operator representing X^T X (Gram matrix)."""

    def __init__(self, X: torch.Tensor):
        super().__init__()

        # Register X as a buffer so it moves with .to(device)
        self.register_buffer("_X", X)
        self._shape = (X.shape[1], X.shape[1])
        # Don't set self._adjoint = self (causes circular reference)
        self.supports_operator_matrix = True

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        return self._X.T @ (self._X @ v)

    @property
    def device(self):
        return self._X.device


def create_ridge_reg_linsys(X: torch.Tensor, y: torch.Tensor, reg: float) -> LinSys:
    """Create ridge regression linear system.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        reg: Regularization parameter

    Returns:
        LinSys representing (X^T X + reg * I) w = X^T y
    """
    A = GramOperator(X)
    b = X.T @ y
    return LinSys(A, b, reg)


def create_pcg_solver(linsys: LinSys, rank: int = 50):
    """Create PCG solver with Nystrom preconditioner.

    Args:
        linsys: Linear system to solve
        rank: Rank for Nystrom approximation

    Returns:
        PCG solver instance
    """
    precond_config = NystromConfig(rank_init=rank, base_damping=0.0)
    return PCG(linsys, config=PCGConfig(preconditioner_config=precond_config))


DATASET_LOADERS = {
    "acsincome": load_acsincome,
    "e2006": load_e2006,
    "realsim": load_realsim,
    "yearpredictionmsd": load_yearpredictionmsd,
}


def run_experiment(
    dataset_name: str,
    device: str,
    num_iters: int = 5000,
    reg: float = 1e-2,
    rank: int = 100,
    seed: int = 42,
):
    """Run a single PCG experiment for a dataset on a specific device.

    Args:
        dataset_name: Name of the dataset to use
        device: Device to run on ('cpu', 'cuda', 'cuda:1', etc.)
        num_iters: Maximum number of iterations
        reg: Ridge regularization parameter
        rank: Rank for Nystrom preconditioner
        seed: Random seed for reproducibility

    Returns:
        dict: Experiment results including status, time, dataset info
    """
    print(f"\n{'=' * 80}")
    print(f"Running experiment: {dataset_name} on {device} (seed={seed})")
    print(f"{'=' * 80}")

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set default dtype
    torch.set_default_dtype(torch.float64)

    # Load dataset
    print("Loading dataset...")
    X_torch, y_torch = DATASET_LOADERS[dataset_name]()
    n_samples, n_features = X_torch.shape

    print(f"Dataset: {dataset_name}")
    print(f"Samples: {n_samples}, Features: {n_features}")

    # Create linear system
    print("Creating linear system...")
    linsys = create_ridge_reg_linsys(X_torch, y_torch, reg=reg)
    linsys = linsys.to(device)

    # Create solver (includes preconditioner construction)
    print(f"Creating PCG solver (rank={rank})...")
    start_construction = time.time()
    solver = create_pcg_solver(linsys, rank=rank)
    construction_time = time.time() - start_construction
    print(f"Solver construction time: {construction_time:.2f}s")

    # Run solver
    print(f"Starting PCG solver on {device}...")
    start_solve = time.time()
    result = solver.solve(
        stopping_criteria=PCGStoppingCriteria(max_iters=num_iters, tol=1e-6)
    )
    solve_time = time.time() - start_solve

    print("Solver finished!")
    print(f"Status: {result.convergence_status}")
    print(f"Construction time: {construction_time:.2f}s")
    print(f"Solve time: {solve_time:.2f}s")
    print(f"Total time: {construction_time + solve_time:.2f}s")
    print(f"Final residual: {result.residual_norm.item():.4e}")

    # Collect results
    results = {
        "dataset": dataset_name,
        "device": device,
        "seed": seed,
        "n_samples": n_samples,
        "n_features": n_features,
        "reg": reg,
        "precond_rank": rank,
        "solver_status": str(result.convergence_status),
        "num_iterations": result.num_iters,
        "construction_time": construction_time,
        "solve_time": solve_time,
        "total_time": construction_time + solve_time,
        "final_residual": result.residual_norm.item(),
    }

    return results


def main():
    """Main function to run a single PCG experiment."""
    parser = argparse.ArgumentParser(
        description="Run PCG solver experiment on a single dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["acsincome", "e2006", "realsim", "yearpredictionmsd"],
        help="Dataset name to run experiment on",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help='Device to run on (e.g., "cpu", "cuda", "cuda:1")',
    )
    parser.add_argument(
        "--num-iters", type=int, default=300, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--reg", type=float, default=1e-2, help="Ridge regularization parameter"
    )
    parser.add_argument(
        "--rank", type=int, default=50, help="Rank for Nystrom preconditioner"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiment_results",
        help="Directory to save results",
    )

    args = parser.parse_args()

    # Create directory for saving results if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Check if CUDA is available when requested
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"ERROR: CUDA is not available but device '{args.device}' was requested.")
        print("Falling back to CPU.")
        args.device = "cpu"

    try:
        # Run experiment
        results = run_experiment(
            args.dataset, args.device, args.num_iters, args.reg, args.rank, args.seed
        )

        # Create DataFrame
        df = pd.DataFrame([results])

        # Save to CSV
        csv_filename = (
            f"pcg_{args.dataset}_{args.device.replace(':', '_')}_seed{args.seed}.csv"
        )
        csv_path = os.path.join(args.output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

    except Exception as e:
        print(f"ERROR running {args.dataset} on {args.device}: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
