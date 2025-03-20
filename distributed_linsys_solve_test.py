from functools import partial

import torch
from torch.multiprocessing import set_start_method

from rlaopt.models import LinSys
from rlaopt.linops import TwoSidedLinOp, DistributedSymmetricLinOp
from rlaopt.preconditioners import NystromConfig
from rlaopt.solvers import PCGConfig


def callback_fn(w, linsys):
    res = torch.linalg.norm(linsys.b - (linsys.A @ w + linsys.reg * w))
    return {"res": res.item()}


def matvec(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return matrix @ x


def rmatvec(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return matrix.T @ x


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    device = torch.device("cuda:0")
    reg = 1e-6
    n = 10000
    n_chunks = 3

    # start workers
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # create a symmetric psd matrix A
    eigvals = torch.arange(1, n + 1, device=device) ** -2.0
    U = torch.randn(n, n, device=device)
    U, _ = torch.linalg.qr(U)
    A = U @ torch.diag(eigvals) @ U.T
    wstar = torch.ones(n, device=device)
    b = A @ wstar + reg * torch.randn(n, device=device)

    # turn A into a DistributedSymmetricLinOp
    A_chunks = A.chunk(n_chunks, dim=0)

    lin_ops = []
    for i, A_chunk in enumerate(A_chunks):
        chunk_device = torch.device(f"cuda:{i}")
        A_chunk = A_chunk.to(chunk_device)
        lin_ops.append(
            TwoSidedLinOp(
                chunk_device,
                A_chunk.shape,
                partial(matvec, matrix=A_chunk),
                partial(rmatvec, matrix=A_chunk),
                partial(matvec, matrix=A_chunk),
                partial(rmatvec, matrix=A_chunk),
            )
        )
    dist_lin_op = DistributedSymmetricLinOp(
        shape=A.shape,
        A=lin_ops,
    )

    # setup linear system and solver
    system = LinSys(A=dist_lin_op, b=b, reg=reg)
    nystrom_config = NystromConfig(rank=100, rho=reg)
    solver_config = PCGConfig(
        precond_config=nystrom_config,
        max_iters=500,
        atol=1e-6,
        rtol=1e-6,
        device=device,
    )

    # solve the system
    system.solve(
        solver_config=solver_config,
        w_init=torch.zeros(n, device=device),
        callback_fn=callback_fn,
        callback_freq=10,
        log_in_wandb=True,
        wandb_init_kwargs={"project": "test_distributed_linsys_solve"},
    )

    dist_lin_op.shutdown()


if __name__ == "__main__":
    main()
