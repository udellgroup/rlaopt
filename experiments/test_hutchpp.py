import torch
import numpy as np
from tqdm import trange
from rlaopt.linops import SymmetricLinOp
from rlaopt.spectral_estimators.trace import hutchinson, hutch_plus_plus


def run_experiment(n=512, trials=1000, seed=42):
    torch.manual_seed(seed)
    errs_hutch = []
    errs_hpp = []
    for _ in trange(trials):
        # random symmetric matrix
        M = torch.randn(n, n)
        A = (M + M.t()) * 0.5
        true_trace = A.trace().item()

        # estimate
        est_h = hutchinson(A, k=256, sketch="rademacher")
        est_p = hutch_plus_plus(A, k=256, sketch="rademacher")

        # errs_hutch.append(abs(est_h - true_trace))
        errs_hutch.append(est_h - true_trace)
        # errs_hpp.append(abs(est_p - true_trace))
        errs_hpp.append(est_p - true_trace)

    return np.array(errs_hutch), np.array(errs_hpp)


if __name__ == "__main__":
    n = 512
    test_matrix = torch.zeros(n, n)
    # test_matrix = torch.randn(n, n)
    # test_matrix = (test_matrix + test_matrix.t()) / 2  # symmetrize
    test_matrix = test_matrix.to("cpu")

    def matvec(x):
        if x.ndim == 1:
            return test_matrix @ x
        else:  # x.ndim == 2
            return test_matrix @ x

    shape = test_matrix.shape
    device = test_matrix.device

    A = SymmetricLinOp(device, shape, matvec, dtype=test_matrix.dtype)

    est = hutchinson(A, k=256, sketch="rademacher")
    hutchpp_est = hutch_plus_plus(A, k=128, sketch="rademacher")
    print(
        f"Hutchinson trace estimate: {est:.3f}, real trace: \
        {test_matrix.trace().item():.3f}"
    )
    print(
        f"Hutch++ trace estimate: {hutchpp_est:.3f}, real trace: \
        {test_matrix.trace().item():.3f}"
    )

    n = 512
    trials = 1000
    errs_hutch, errs_hpp = run_experiment(n, trials)

    # Basic stats
    for name, errs in [("Hutchinson", errs_hutch), ("Hutch++", errs_hpp)]:
        print(f"{name}:")
        print(f"  mean err   = {errs.mean():.4f}")
        print(f"  median err = {np.median(errs):.4f}")
        print(f"  std err    = {errs.std():.4f}")
        print()
