from functools import partial

from pykeops.torch import LazyTensor
import torch
from torch.multiprocessing import Pool, set_start_method

from rlaopt.models import LinSys
from rlaopt.utils import TwoSidedLinOp, DistributedSymmetricLinOp
from rlaopt.preconditioners import NystromConfig
from rlaopt.solvers import PCGConfig


# helper function to initialize worker devices
def initialize_worker(device_id: int, n_devices: int):
    if torch.cuda.is_available():
        device = device_id % n_devices
        torch.cuda.set_device(device)
    else:
        print(f"Worker {device_id} using CPU")


def callback_fn(w, linsys):
    res = torch.linalg.norm(linsys.b - (linsys.A @ w + linsys.reg * w))
    return {"res": res.item()}


def matvec(x: torch.Tensor, A, Ab, sigma):
    # Compute the kernel matrix
    Ab_lazy = LazyTensor(Ab[:, None, :])
    A_lazy = LazyTensor(A[None, :, :])
    D = ((Ab_lazy - A_lazy) ** 2).sum(dim=2)
    Kb = (-D / (2 * sigma**2)).exp()
    return Kb @ x


def rmatvec(x: torch.Tensor, A, Ab, sigma):
    # Compute the kernel matrix
    # Ab_lazy = LazyTensor(Ab[None, :, :])
    # A_lazy = LazyTensor(A[:, None, :])
    # D = ((A_lazy - Ab_lazy) ** 2).sum(dim=2)
    # KbT = (-D / (2 * sigma ** 2)).exp()
    # return KbT @ x
    Ab_lazy = LazyTensor(Ab[:, None, :])
    A_lazy = LazyTensor(A[None, :, :])
    D = ((Ab_lazy - A_lazy) ** 2).sum(dim=2)
    Kb = (-D / (2 * sigma**2)).exp()
    return Kb.T @ x


# NOTE(pratik): this class does not work because the LazyTensor has to be made within
# the worker. The easisest way to do this is to make the LazyTensor in the matvec and
# rmatvec functions.
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
    def __init__(self, A, Ab, sigma):
        self.A = A
        self.Ab = Ab
        self.sigma = sigma

        super().__init__(
            device=A.device,
            shape=torch.Size((Ab.shape[0], A.shape[0])),
            matvec=partial(matvec, A=A, Ab=Ab, sigma=sigma),
            rmatvec=partial(rmatvec, A=A, Ab=Ab, sigma=sigma),
            matmat=partial(matvec, A=A, Ab=Ab, sigma=sigma),
            rmatmat=partial(rmatvec, A=A, Ab=Ab, sigma=sigma),
        )


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    device = torch.device("cuda:0")
    reg = 1e-6
    n = 30000
    d = 10
    sigma = 1.0
    n_chunks = 3
    n_devices = 3

    # start workers
    try:
        set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # create the pool with device initializers
    pool = Pool(
        processes=n_chunks,
        initializer=initialize_worker,
        initargs=(list(range(n_chunks))[0], n_devices),
    )

    # generate synthetic data
    A = torch.randn(n, d, device=device) / (n**0.5)
    b = torch.randn(n, device=device)

    # get DistributedSymmetricLinOp for kernel matrix
    A_chunk_idx = torch.arange(A.shape[0]).chunk(n_chunks)
    A_copies = [A.to(f"cuda:{i}") for i in range(n_chunks)]

    lin_ops = []
    for chunk_idx, A_copy in zip(A_chunk_idx, A_copies):
        lin_ops.append(RBFLinearOperator(A_copy, A_copy[chunk_idx], sigma))
    dist_lin_op = DistributedSymmetricLinOp(
        shape=torch.Size((A.shape[0], A.shape[0])),
        A=lin_ops,
        pool=pool,
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
        wandb_init_kwargs={"project": "test_distributed_krr_linsys_solve"},
    )


if __name__ == "__main__":
    main()
