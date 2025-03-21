from functools import partial

from pykeops.torch import LazyTensor
import torch

from rlaopt.models import LinSys
from rlaopt.linops import DistributionMode, LinOp, SymmetricLinOp, DistributedLinOp
from rlaopt.preconditioners import NystromConfig
from rlaopt.solvers import SAPConfig, SAPAccelConfig


def callback_fn(w, linsys):
    res = torch.linalg.norm(linsys.b - (linsys.A @ w + linsys.reg * w))
    return {"res": res.item()}


def matvec(x: torch.Tensor, A, chunk_idx, sigma):
    # Compute the kernel matrix
    Ab_lazy = LazyTensor(A[chunk_idx][:, None, :])
    A_lazy = LazyTensor(A[None, :, :])
    D = ((Ab_lazy - A_lazy) ** 2).sum(dim=2)
    Kb = (-D / (2 * sigma**2)).exp()
    return Kb @ x


def row_matvec(x: torch.Tensor, A_blk, A, sigma):
    # Compute the kernel matrix
    Ab_lazy = LazyTensor(A_blk[:, None, :])
    A_lazy = LazyTensor(A[None, :, :])
    D = ((Ab_lazy - A_lazy) ** 2).sum(dim=2)
    Kb = (-D / (2 * sigma**2)).exp()
    return Kb @ x


def row_oracle(blk: torch.Tensor, A, A_chunks, sigma):
    lin_ops = []
    for A_chunk in A_chunks:
        matvec_fn = partial(
            row_matvec, A_blk=A[blk].to(A_chunk.device), A=A_chunk, sigma=sigma
        )
        lin_ops.append(
            LinOp(
                device=A_chunk.device,
                shape=torch.Size((blk.shape[0], A_chunk.shape[0])),
                matvec=matvec_fn,
                matmat=matvec_fn,
            )
        )
    return DistributedLinOp(
        shape=torch.Size(
            (blk.shape[0], sum([A_chunk.shape[0] for A_chunk in A_chunks]))
        ),
        A=lin_ops,
        distribution_mode=DistributionMode.COLUMN,
    )


def blk_oracle(blk: torch.Tensor, A, sigma, device):
    matvec_fn = partial(
        matvec, A=A[blk].to(device), chunk_idx=torch.arange(blk.shape[0]), sigma=sigma
    )
    return SymmetricLinOp(
        device=device,
        shape=torch.Size((blk.shape[0], blk.shape[0])),
        matvec=matvec_fn,
        matmat=matvec_fn,
    )


class RBFLinearOperator(SymmetricLinOp):
    def __init__(self, A, sigma):
        matvec_fn = partial(
            matvec, A=A, chunk_idx=torch.arange(A.shape[0]), sigma=sigma
        )
        super().__init__(
            device=A.device,
            shape=torch.Size((A.shape[0], A.shape[0])),
            matvec=matvec_fn,
            matmat=matvec_fn,
        )


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    loading_device = torch.device("cpu")
    compute_device = torch.device("cuda:0")
    reg = 1e-8
    n = 100000
    d = 100
    sigma = 1.0
    n_chunks = 5

    # generate synthetic data
    A = torch.randn(n, d, device=loading_device) / (n**0.5)
    b = torch.randn(n, device=loading_device)

    # chunk the data across devices
    A_chunks = torch.chunk(A, n_chunks, dim=0)
    A_chunks = [A_chunk.to(f"cuda:{i}") for i, A_chunk in enumerate(A_chunks)]

    # get row oracles for kernel matrix
    row_oracle_fn = partial(row_oracle, A=A, A_chunks=A_chunks, sigma=sigma)

    # get block oracles for kernel matrix
    blk_oracle_fn = partial(blk_oracle, A=A, sigma=sigma, device=compute_device)

    # get linear operator for kernel matrix
    lin_op = RBFLinearOperator(A.to(compute_device), sigma)

    # setup linear system and solver
    system = LinSys(
        A=lin_op,
        b=b.to(compute_device),
        reg=reg,
        A_row_oracle=row_oracle_fn,
        A_blk_oracle=blk_oracle_fn,
    )
    nystrom_config = NystromConfig(rank=100, rho=reg)
    accel_config = SAPAccelConfig(mu=reg, nu=100.0)
    solver_config = SAPConfig(
        precond_config=nystrom_config,
        max_iters=1000,
        atol=1e-6,
        rtol=1e-6,
        blk_sz=n // 100,
        accel_config=accel_config,
        device=compute_device,
    )

    # solve the system
    system.solve(
        solver_config=solver_config,
        w_init=torch.zeros(n, device=compute_device),
        callback_fn=callback_fn,
        callback_freq=10,
        log_in_wandb=True,
        wandb_init_kwargs={"project": "test_distributed_krr_linsys_askotch_solve"},
    )


if __name__ == "__main__":
    main()
