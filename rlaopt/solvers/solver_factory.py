from typing import TYPE_CHECKING

import torch

from rlaopt.solvers.pcg import PCG
from rlaopt.solvers.sap import SAP
from rlaopt.solvers.configs import SolverConfig, PCGConfig, SAPConfig

if TYPE_CHECKING:
    from rlaopt.models import Model, LinSys  # Import only for type hints


def _get_pcg(model: "LinSys", w_init: torch.Tensor, pcg_config: PCGConfig):
    return PCG(
        system=model,
        w_init=w_init,
        device=pcg_config.device,
        precond_config=pcg_config.precond_config,
    )


def _get_sap(model: "LinSys", w_init: torch.Tensor, sap_config: SAPConfig):
    return SAP(
        system=model,
        w_init=w_init,
        device=sap_config.device,
        precond_config=sap_config.precond_config,
        blk_sz=sap_config.blk_sz,
        accel=sap_config.accel,
        accel_config=sap_config.accel_config,
        power_iters=sap_config.power_iters,
    )


def _get_solver(model: "Model", w_init: torch.Tensor, solver_config: SolverConfig):
    # Get the class of solver_config
    solver_config_class = solver_config.__class__

    if solver_config_class == PCGConfig:
        return _get_pcg(model, w_init, solver_config)
    elif solver_config_class == SAPConfig:
        return _get_sap(model, w_init, solver_config)
