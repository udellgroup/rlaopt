from sketchyopts import preconditioner
from sketchyopts import solver
from sketchyopts import errors

rand_nystrom_approx = preconditioner.rand_nystrom_approx
nystrom_pcg = solver.nystrom_pcg
InputDimError = errors.InputDimError
MatrixNotSquareError = errors.MatrixNotSquareError

__all__ = (
    "rand_nystrom_approx",
    "nystrom_pcg",
    "InputDimError",
    "MatrixNotSquareError",
)
