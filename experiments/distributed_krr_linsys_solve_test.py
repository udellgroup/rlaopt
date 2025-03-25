import torch

from rlaopt.models import LinSys
from rlaopt.kernels import RBFLinOp, DistributedRBFLinOp
from rlaopt.preconditioners import NystromConfig
from rlaopt.solvers import PCGConfig


def callback_fn(w, linsys):
    res = torch.linalg.norm(linsys.b - (linsys.A @ w + linsys.reg * w))
    return {"res": res.item()}


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    device = torch.device("cuda:0")
    reg = 1e-8
    n = 100000
    d = 100
    sigma = 1.0
    n_chunks = 5

    # generate synthetic data
    A = torch.randn(n, d, device=device) / (n**0.5)
    b = torch.randn(n, device=device)

    devices = set([torch.device(f"cuda:{i}") for i in range(n_chunks)])

    dist_lin_op = DistributedRBFLinOp(
        A=A,
        sigma=sigma,
        devices=devices,
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
    lin_op = RBFLinOp(
        A=A,
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
