from functools import partial

from pykeops.torch import LazyTensor
import torch
from torch.multiprocessing import set_start_method

from rlaopt.models import LinSys
from rlaopt.linops import TwoSidedLinOp, DistributedSymmetricLinOp
from rlaopt.preconditioners import NystromConfig
from rlaopt.solvers import PCGConfig


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


def rmatvec(x: torch.Tensor, A, chunk_idx, sigma):
    # Compute the kernel matrix
    Ab_lazy = LazyTensor(A[chunk_idx][None, :, :])
    A_lazy = LazyTensor(A[:, None, :])
    D = ((A_lazy - Ab_lazy) ** 2).sum(dim=2)
    KbT = (-D / (2 * sigma**2)).exp()
    return KbT @ x


# NOTE(pratik): this class does not work because the LazyTensor has to be made within
# the worker. The easiest way to do this is to make the LazyTensor in the matvec and
# rmatvec functions, which are used by the worker in the distributed linear operators.
# class RBFLinearOperator(TwoSidedLinOp):
#     def __init__(self, A, Ab, sigma):
#         # Compute the kernel matrix
#         Ab_lazy = LazyTensor(Ab[:, None, :])
#         A_lazy = LazyTensor(A[None, :, :])
#         D = ((Ab_lazy - A_lazy) ** 2).sum(dim=2)
#         Kb = (-D / (2 * sigma ** 2)).exp()

#         super().__init__(
#             device=A.device,
#             shape=torch.Size(Kb.shape),
#             matvec=partial(matvec, matrix=Kb),
#             rmatvec=partial(rmatvec, matrix=Kb),
#             matmat=partial(matvec, matrix=Kb),
#             rmatmat=partial(rmatvec, matrix=Kb),
#         )


class RBFLinearOperator(TwoSidedLinOp):
    def __init__(self, A, chunk_idx, sigma):
        super().__init__(
            device=A.device,
            shape=torch.Size((chunk_idx.shape[0], A.shape[0])),
            matvec=partial(matvec, A=A, chunk_idx=chunk_idx, sigma=sigma),
            rmatvec=partial(rmatvec, A=A, chunk_idx=chunk_idx, sigma=sigma),
            matmat=partial(matvec, A=A, chunk_idx=chunk_idx, sigma=sigma),
            rmatmat=partial(rmatvec, A=A, chunk_idx=chunk_idx, sigma=sigma),
        )


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    device = torch.device("cuda:0")
    reg = 1e-8
    n = 100000
    d = 10
    sigma = 1.0
    n_chunks = 3

    # start workers
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # generate synthetic data
    A = torch.randn(n, d, device=device) / (n**0.5)
    b = torch.randn(n, device=device)

    # get DistributedSymmetricLinOp for kernel matrix
    A_chunk_idx = torch.arange(A.shape[0]).chunk(n_chunks)

    lin_ops = []
    for i, chunk_idx in enumerate(A_chunk_idx):
        lin_ops.append(RBFLinearOperator(A.to(f"cuda:{i}"), chunk_idx, sigma))
    dist_lin_op = DistributedSymmetricLinOp(
        shape=torch.Size((A.shape[0], A.shape[0])),
        A=lin_ops,
    )

    # setup linear system and solver
    system = LinSys(A=dist_lin_op, b=b, reg=reg)
    nystrom_config = NystromConfig(rank=100, rho=reg)
    solver_config = PCGConfig(
        precond_config=nystrom_config,
        max_iters=60,
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
        wandb_init_kwargs={"project": "test_distributed_krr_linsys_solve"},
    )

    dist_lin_op.shutdown()

    # now do it without distribution
    lin_op = RBFLinearOperator(
        A=A,
        chunk_idx=torch.arange(A.shape[0]),
        sigma=sigma,
    )
    system = LinSys(A=lin_op, b=b, reg=reg)
    system.solve(
        solver_config=solver_config,
        w_init=torch.zeros(n, device=device),
        callback_fn=callback_fn,
        callback_freq=10,
        log_in_wandb=True,
        wandb_init_kwargs={"project": "test_distributed_krr_linsys_solve"},
    )


if __name__ == "__main__":
    main()
