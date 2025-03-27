import time
from typing import Callable, Optional, List, Tuple

import numpy as np
import scipy.sparse as sp
import torch

from rlaopt.sparse import SparseCSRTensor


def get_devices():
    """Return a list of available devices to test."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")
    return devices


def format_device_name(device_str):
    """Format device name for display."""
    if device_str == "cpu":
        return "CPU"
    else:
        gpu_id = device_str.split(":")[-1]
        return f"GPU {gpu_id}"


def load_sparse_matrix(file_path, precision):
    """Load sparse matrix from npz file."""
    return sp.load_npz(file_path).astype(precision)


def time_and_verify_operation(
    operation: Callable,
    reference_operation: Callable,
    device_str: str,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Tuple[float, bool]:
    """Perform operation, time it, and verify against reference."""
    start = time.time()
    result = operation()
    elapsed = time.time() - start

    ref_start = time.time()
    ref_result = reference_operation()
    ref_elapsed = time.time() - ref_start

    # Convert reference result to torch tensor on the right device if needed
    if isinstance(ref_result, np.ndarray):
        ref_result = torch.tensor(ref_result, device=device_str)

    # Verify results match
    is_correct = torch.allclose(result, ref_result, rtol=rtol, atol=atol)

    return elapsed, ref_elapsed, is_correct


def run_row_slicing_test(X_torch, device_str):
    """Test for CSR row slicing."""
    device = torch.device(device_str)
    idx = torch.tensor(np.arange(1024), dtype=torch.int64, device=device)
    b = torch.randn(X_torch.shape[1], 10, device=device)

    def operation():
        return X_torch[idx] @ b

    def reference_operation():
        return (X_torch @ b)[idx]

    _, _, is_correct = time_and_verify_operation(
        operation, reference_operation, device_str
    )

    print("  CSR Row Slicing:")
    print(f"    Correctness: {'✓' if is_correct else '✗'}")

    return is_correct


def run_csc_matvec_test(X_torch, X, device_str):
    """Test for CSC matrix-vector multiply."""
    device = torch.device(device_str)
    d = torch.randn(X_torch.shape[0], device=device)
    d_np = d.cpu().numpy()

    def operation():
        return X_torch.T @ d

    def reference_operation():
        return X.T @ d_np

    elapsed, ref_elapsed, is_correct = time_and_verify_operation(
        operation, reference_operation, device_str
    )

    print("  CSC Matrix-Vector Multiply:")
    print(f"    Time (extension): {elapsed:.6f}s")
    print(f"    Time (scipy): {ref_elapsed:.6f}s")
    print(f"    Speedup: {ref_elapsed/elapsed:.2f}x")
    print(f"    Correctness: {'✓' if is_correct else '✗'}")

    return is_correct


def run_csc_matmat_test(X_torch, X, device_str, cols):
    """Test for CSC matrix-matrix multiply."""
    device = torch.device(device_str)
    D = torch.randn(X_torch.shape[0], cols, device=device)
    D_np = D.cpu().numpy()

    def operation():
        return X_torch.T @ D

    def reference_operation():
        return X.T @ D_np

    elapsed, ref_elapsed, is_correct = time_and_verify_operation(
        operation, reference_operation, device_str
    )

    print(f"  CSC Matrix-Matrix Multiply ({cols} columns):")
    print(f"    Time (extension): {elapsed:.6f}s")
    print(f"    Time (scipy): {ref_elapsed:.6f}s")
    print(f"    Speedup: {ref_elapsed/elapsed:.2f}x")
    print(f"    Correctness: {'✓' if is_correct else '✗'}")

    return is_correct


def run_tests(
    file_path: str = "yelp_train.npz",
    torch_precision: torch.dtype = torch.float64,
    devices: Optional[List[str]] = None,
):
    """Run all tests for sparse tensor operations."""
    # Set up precision
    np_precision = np.float32 if torch_precision == torch.float32 else np.float64
    torch.set_default_dtype(torch_precision)

    # Load data
    X = load_sparse_matrix(file_path, np_precision)

    if devices is None:
        devices = get_devices()

    all_passed = True

    for device_str in devices:
        device = torch.device(device_str)
        device_name = format_device_name(device_str)
        print(f"\n{'='*50}")
        print(f"Running tests on {device_name}")
        print(f"{'='*50}")

        # Create sparse tensors
        X_torch = SparseCSRTensor(data=X, device=device)

        print(f"\nMatrix shape: {X_torch.shape}")
        print(f"Precision: {torch_precision}")

        # Run tests
        row_slicing_passed = run_row_slicing_test(X_torch, device_str)
        csc_matvec_passed = run_csc_matvec_test(X_torch, X, device_str)
        csc_matmat_passed = run_csc_matmat_test(X_torch, X, device_str, cols=32)

        # Run larger matrix-matrix test if on GPU
        if "cuda" in device_str:
            csc_large_matmat_passed = run_csc_matmat_test(
                X_torch, X, device_str, cols=128
            )
            test_results = [
                row_slicing_passed,
                csc_matvec_passed,
                csc_matmat_passed,
                csc_large_matmat_passed,
            ]
        else:
            test_results = [row_slicing_passed, csc_matvec_passed, csc_matmat_passed]

        # Summary for this device
        print(f"\nSummary for {device_name}:")
        print(f"  Tests passed: {sum(test_results)}/{len(test_results)}")

        all_passed = all_passed and all(test_results)

    # Overall summary
    print("\n" + "=" * 50)
    print(f"Overall test result: {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 50)

    return all_passed


def main():
    """Main function to run the tests."""
    print("\nSparse Tensor Operations Test Suite")
    print("----------------------------------")

    # Test with double precision (default)
    run_tests(torch_precision=torch.float64, devices=["cpu", "cuda:1"])

    # Uncomment to test with single precision
    # run_tests(torch_precision=torch.float32)


if __name__ == "__main__":
    main()
