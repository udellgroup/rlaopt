from sketchyopts import preconditioner
from sketchyopts import solver
from sketchyopts import util
from sketchyopts import errors

rand_nystrom_approx = preconditioner.rand_nystrom_approx
NystromPrecondState = preconditioner.NystromPrecondState
update_nystrom_precond = preconditioner.update_nystrom_precond
scale_by_nystrom_precond = preconditioner.scale_by_nystrom_precond
nystrom_pcg = solver.nystrom_pcg
sketchysgd = solver.sketchysgd
LinearOperator = util.LinearOperator
shareble_state_named_chain = util.shareble_state_named_chain
scale_by_ref_learning_rate = util.scale_by_ref_learning_rate
InputDimError = errors.InputDimError
MatrixNotSquareError = errors.MatrixNotSquareError

__all__ = (
    "rand_nystrom_approx",
    "NystromPrecondState",
    "update_nystrom_precond",
    "scale_by_nystrom_precond",
    "nystrom_pcg",
    "sketchysgd",
    "LinearOperator",
    "shareble_state_named_chain",
    "scale_by_ref_learning_rate",
    "InputDimError",
    "MatrixNotSquareError",
)
