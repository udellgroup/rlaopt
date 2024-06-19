from sketchyopts import preconditioner
from sketchyopts import solver
from sketchyopts import base
from sketchyopts import util
from sketchyopts import errors

rand_nystrom_approx = preconditioner.rand_nystrom_approx
nystrom_pcg = solver.nystrom_pcg
SketchySGD = solver.SketchySGD
SketchySVRG = solver.SketchySVRG
LinearOperator = base.LinearOperator
HessianLinearOperator = base.HessianLinearOperator
generate_random_batch = util.generate_random_batch
InputDimError = errors.InputDimError
MatrixNotSquareError = errors.MatrixNotSquareError

__all__ = (
    "rand_nystrom_approx",
    "nystrom_pcg",
    "SketchySGD",
    "SketchySVRG",
    "LinearOperator",
    "HessianLinearOperator",
    "generate_random_batch",
    "InputDimError",
    "MatrixNotSquareError",
)
