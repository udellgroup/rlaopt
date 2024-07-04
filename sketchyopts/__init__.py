from sketchyopts import base, errors, preconditioner, solver, util

rand_nystrom_approx = preconditioner.rand_nystrom_approx
nystrom_pcg = solver.nystrom_pcg
SketchySGD = solver.SketchySGD
SketchySGDState = solver.SketchySGDState
SketchySVRG = solver.SketchySVRG
SketchySVRGState = solver.SketchySVRGState
SketchySAGA = solver.SketchySAGA
SketchySAGAState = solver.SketchySAGAState
SketchyKatyusha = solver.SketchyKatyusha
SketchyKatyushaState = solver.SketchyKatyushaState
SolverState = base.SolverState
PromiseSolver = base.PromiseSolver
LinearOperator = base.LinearOperator
HessianLinearOperator = base.HessianLinearOperator
ravel_tree = util.ravel_tree
tree_flatten = util.tree_flatten
tree_unflatten = util.tree_unflatten
tree_add = util.tree_add
tree_sub = util.tree_sub
tree_scalar_mul = util.tree_scalar_mul
tree_add_scalar_mul = util.tree_add_scalar_mul
tree_l2_norm = util.tree_l2_norm
InputDimError = errors.InputDimError
MatrixNotSquareError = errors.MatrixNotSquareError

__all__ = (
    "rand_nystrom_approx",
    "nystrom_pcg",
    "SketchySGD",
    "SketchySGDState",
    "SketchySVRG",
    "SketchySVRGState",
    "SketchySAGA",
    "SketchySAGAState",
    "SketchyKatyusha",
    "SketchyKatyushaState",
    "SolverState",
    "PromiseSolver",
    "LinearOperator",
    "HessianLinearOperator",
    "ravel_tree",
    "tree_flatten",
    "tree_unflatten",
    "tree_add",
    "tree_sub",
    "tree_scalar_mul",
    "tree_add_scalar_mul",
    "tree_l2_norm",
    "InputDimError",
    "MatrixNotSquareError",
)
