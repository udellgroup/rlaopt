import torch

from rlaopt.models import LinSys
from rlaopt.kernels import DistributedRBFLinOp
from rlaopt.preconditioners import NystromConfig
from rlaopt.solvers import SAPConfig, SAPAccelConfig


def callback_fn(W, linsys):
    res = torch.linalg.norm(linsys.B - (linsys.A @ W + linsys.reg * W), dim=0, ord=2)
    return {"res": res.item()}


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    loading_device = torch.device("cpu")
    compute_device = torch.device("cuda:0")
    n = 1000000
    d = 100
    sigma = 1.0
    reg = 1e-8 * n
    n_chunks = 5

    # generate synthetic data
    A = torch.randn(n, d, device=loading_device) / (n**0.5)
    b = torch.randn(n, device=loading_device)

    # get linear operator for kernel matrix
    lin_op = DistributedRBFLinOp(
        A=A,
        kernel_params={"lengthscale": sigma},
        devices=set([torch.device(f"cuda:{i}") for i in range(n_chunks)]),
    )

    # setup linear system and solver
    system = LinSys(
        A=lin_op,
        B=b.to(compute_device),
        reg=reg,
        A_row_oracle=lin_op.row_oracle,
        A_blk_oracle=lin_op.blk_oracle,
    )
    nystrom_config = NystromConfig(rank=1000, rho=reg)
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
        W_init=torch.zeros(n, 1, device=compute_device),
        callback_fn=callback_fn,
        callback_freq=100,
        log_in_wandb=True,
        wandb_init_kwargs={"project": "test_distributed_krr_linsys_askotch_solve"},
    )


if __name__ == "__main__":
    main()
