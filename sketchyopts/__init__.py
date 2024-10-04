from sketchyopts import base, errors, preconditioner, prox, solver, util

rand_nystrom_approx = preconditioner.rand_nystrom_approx
nystrom_pcg = solver.nystrom_pcg
NysADMM = solver.NysADMM
NysADMMState = solver.NysADMMState
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
AddLinearOperator = base.AddLinearOperator
prox_const = prox.prox_const
prox_l1 = prox.prox_l1
prox_nonnegative_l1 = prox.prox_nonnegative_l1
prox_elastic_net = prox.prox_elastic_net
prox_l2 = prox.prox_l2
prox_l2_squared = prox.prox_l2_squared
prox_nonnegative_l2_squared = prox.prox_nonnegative_l2_squared
prox_nonnegative = prox.prox_nonnegative
prox_box = prox.prox_box
prox_hyperplane = prox.prox_hyperplane
prox_halfspace = prox.prox_halfspace
ravel_tree = util.ravel_tree
tree_map = util.tree_map
tree_flatten = util.tree_flatten
tree_unflatten = util.tree_unflatten
tree_add = util.tree_add
tree_sub = util.tree_sub
tree_scalar_mul = util.tree_scalar_mul
tree_add_scalar_mul = util.tree_add_scalar_mul
tree_vdot = util.tree_vdot
tree_l2_norm = util.tree_l2_norm
tree_zeros_like = util.tree_zeros_like
tree_ones_like = util.tree_ones_like
default_floating_dtype = util.default_floating_dtype
default_integer_dtype = util.default_integer_dtype
inexact_asarray = util.inexact_asarray
integer_asarray = util.integer_asarray
InputDimError = errors.InputDimError
MatrixNotSquareError = errors.MatrixNotSquareError

__all__ = (
    "rand_nystrom_approx",
    "nystrom_pcg",
    "NysADMM",
    "NysADMMState",
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
    "AddLinearOperator",
    "prox_const",
    "prox_l1",
    "prox_nonnegative_l1",
    "prox_elastic_net",
    "prox_l2",
    "prox_l2_squared",
    "prox_nonnegative_l2_squared",
    "prox_nonnegative",
    "prox_box",
    "prox_hyperplane",
    "prox_halfspace",
    "ravel_tree",
    "tree_map",
    "tree_flatten",
    "tree_unflatten",
    "tree_add",
    "tree_sub",
    "tree_scalar_mul",
    "tree_add_scalar_mul",
    "tree_vdot",
    "tree_l2_norm",
    "tree_zeros_like",
    "tree_ones_like",
    "default_floating_dtype",
    "default_integer_dtype",
    "inexact_asarray",
    "integer_asarray",
    "InputDimError",
    "MatrixNotSquareError",
)
