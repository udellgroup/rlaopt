"""Run ADMM solver with per-iteration timing for CPU vs GPU comparison."""

import argparse
import json
import os
import time

import pandas as pd
import torch
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import StandardScaler

from rlaopt.atoms import Box, ElasticNet, SumSquares
from rlaopt.expression import Variable
from rlaopt.linalg import NystromConfig
from rlaopt.solvers import ADMM, ADMMConfig


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


def create_constrained_elastic_net(X_torch, y_torch):
    """Create constrained elastic net optimization problem."""
    X_T_y = X_torch.T @ y_torch
    lambd = 0.1 * torch.linalg.norm(X_T_y, ord=float("inf")) / X_torch.shape[0]
    w = Variable((X_torch.shape[1],), name="w")
    b = Variable((1,), name="b")
    obj = SumSquares(X_torch @ w + b - y_torch) * (0.5 / X_torch.shape[0]) + ElasticNet(
        w, l1_scaling=lambd, l2_scaling=lambd
    )
    constraints = Box(w, lower=0.0, upper=1.0)
    return obj, constraints, w, b


def combine_obj_constraints(obj, constraints):
    """Combine objective and constraints."""
    if constraints is None:
        return obj
    return obj + constraints


def create_admm_solver(loss, n_features: int):
    """Create ADMM solver with Nystrom preconditioner."""
    precond_config = NystromConfig(
        rank_init=min(50, n_features // 10),
        base_damping=0.0,
    )
    return ADMM(loss, config=ADMMConfig(rho=1e0, preconditioner_config=precond_config))


DATASET_LOADERS = {
    "acsincome": load_acsincome,
    "e2006": load_e2006,
    "realsim": load_realsim,
    "yearpredictionmsd": load_yearpredictionmsd,
}


def run_admm_with_iteration_timing(
    dataset_name: str,
    device: str,
    num_iters: int = 100,
    seed: int = 42,
):
    """Run ADMM and record timing and residual norms at each iteration.

    Args:
        dataset_name: Name of the dataset to use
        device: Device to run on ('cpu', 'cuda', 'cuda:1', etc.)
        num_iters: Number of iterations to run
        seed: Random seed for reproducibility

    Returns:
        dict: Results including per-iteration timing and residual norms
    """
    print(f"\n{'=' * 80}")
    print(f"Running ADMM iteration timing: {dataset_name} on {device} (seed={seed})")
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

    # Create optimization problem
    print("Creating optimization problem...")
    obj, constraints, w, b = create_constrained_elastic_net(X_torch, y_torch)
    loss = combine_obj_constraints(obj, constraints)
    loss = loss.to(device)

    # Create solver
    print("Creating ADMM solver...")
    start_construction = time.time()
    solver = create_admm_solver(loss, n_features)
    construction_time = time.time() - start_construction
    print(f"Solver construction time: {construction_time:.2f}s")

    # Initialize solver state
    print(f"Starting ADMM solver on {device} for {num_iters} iterations...")

    variable_values = loss.variable_values
    state = solver.init_state(variable_values)

    # Track iteration data
    iteration_times = []
    cumulative_times = []
    primal_residual_norms = []
    dual_residual_norms = []
    iterations = []

    cumulative_time = 0.0

    # Record initial state
    iterations.append(0)
    primal_residual_norms.append(state.primal_residual_norm)
    dual_residual_norms.append(state.dual_residual_norm)
    iteration_times.append(0.0)
    cumulative_times.append(0.0)

    print(
        f"Iter 0: primal_res = {state.primal_residual_norm:.4e}, "
        f"dual_res = {state.dual_residual_norm:.4e}"
    )

    # Run iterations manually
    start_solve = time.time()
    for iter_num in range(1, num_iters + 1):
        iter_start = time.time()

        # Perform one iteration step
        variable_values, state = solver.step(variable_values, state)

        # Synchronize if using CUDA to get accurate timing
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        iter_time = time.time() - iter_start
        cumulative_time += iter_time

        # Record iteration data
        iterations.append(iter_num)
        primal_residual_norms.append(state.primal_residual_norm)
        dual_residual_norms.append(state.dual_residual_norm)
        iteration_times.append(iter_time)
        cumulative_times.append(cumulative_time)

        # Print progress every 10 iterations
        if iter_num % 10 == 0 or iter_num == num_iters:
            print(
                f"Iter {iter_num}: primal_res = {state.primal_residual_norm:.4e}, "
                f"dual_res = {state.dual_residual_norm:.4e}, "
                f"iter_time = {iter_time:.4f}s, cumulative = {cumulative_time:.2f}s"
            )

    total_solve_time = time.time() - start_solve

    print("\nSolver finished!")
    print(f"Total iterations: {num_iters}")
    print(f"Construction time: {construction_time:.2f}s")
    print(f"Total solve time: {total_solve_time:.2f}s")
    print(f"Final primal residual: {state.primal_residual_norm:.4e}")
    print(f"Final dual residual: {state.dual_residual_norm:.4e}")

    # Compile results
    results = {
        "dataset": dataset_name,
        "device": device,
        "seed": seed,
        "n_samples": n_samples,
        "n_features": n_features,
        "total_iterations": num_iters,
        "construction_time": construction_time,
        "total_solve_time": total_solve_time,
        "final_primal_residual": state.primal_residual_norm,
        "final_dual_residual": state.dual_residual_norm,
        "iterations": iterations,
        "primal_residual_norms": primal_residual_norms,
        "dual_residual_norms": dual_residual_norms,
        "iteration_times": iteration_times,
        "cumulative_times": cumulative_times,
    }

    return results


def main():
    """Main function to run ADMM with iteration timing."""
    parser = argparse.ArgumentParser(
        description="Run ADMM solver with per-iteration timing"
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
        "--num-iters", type=int, default=100, help="Number of iterations to run"
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
        results = run_admm_with_iteration_timing(
            args.dataset, args.device, args.num_iters, args.seed
        )

        # Save to JSON (for detailed iteration data)
        json_filename = f"admm_iterations_{args.dataset}_{args.device.replace(':', '_')}_seed{args.seed}.json"
        json_path = os.path.join(args.output_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {json_path}")

    except Exception as e:
        print(f"ERROR running {args.dataset} on {args.device}: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
