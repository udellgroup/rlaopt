import torch

from rlaopt.models import LinSys
from rlaopt.kernels import DistributedRBFLinOp
from rlaopt.preconditioners import NystromConfig
from rlaopt.solvers import SAPConfig, SAPAccelConfig


def main():
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(0)

    # experiment parameters
    n = 1000000
    d = 300
    k = 10
    sigma = 1.0
    reg = 1e-8 * n
    # devices = [torch.device("cuda:2")]
    devices = [torch.device("cuda:3"), torch.device("cuda:4")]

    # generate synthetic data
    A = torch.randn(n, d, device=devices[0]) / (d**0.5)
    b = torch.randn(n, k, device=devices[0])

    # get linear operator for kernel matrix
    lin_op = DistributedRBFLinOp(
        A=A,
        # kernel_params={"lengthscale": sigma},
        kernel_params={"lengthscale": sigma * torch.ones(d, device=devices[0])},
        devices=set(devices),
    )

    # setup linear system and solver
    system = LinSys(
        A=lin_op,
        B=b.to(devices[0]),
        reg=reg,
        A_row_oracle=lin_op.row_oracle,
        A_blk_oracle=lin_op.blk_oracle,
    )
    nystrom_config = NystromConfig(rank=100, rho=reg)
    accel_config = SAPAccelConfig(mu=reg, nu=100.0)
    solver_config = SAPConfig(
        precond_config=nystrom_config,
        max_iters=300,
        atol=1e-6,
        rtol=1e-6,
        blk_sz=n // 100,
        accel_config=accel_config,
        device=devices[0],
    )

    # solve the system
    system.solve(
        solver_config=solver_config,
        W_init=torch.zeros(n, k, device=devices[0]),
        callback_freq=100,
        log_in_wandb=True,
        wandb_init_kwargs={
            "project": "test_distributed_krr_linsys_askotch_solve_v2",
            "config": {
                "n": n,
                "d": d,
                "k": k,
                "reg": reg,
                "sigma": sigma,
                "devices": devices,
                "dtype": dtype,
            },
        },
    )


if __name__ == "__main__":
    main()
