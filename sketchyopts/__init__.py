from sketchyopts import (
    base,
    errors,
    linear_solve,
    nystrom,
    operator,
    prox,
    sketching,
    util,
)

from .solver import NysADMM as NysADMM
from .solver import NysADMMState as NysADMMState
from .solver import PromiseSolver as PromiseSolver
from .solver import SketchyKatyusha as SketchyKatyusha
from .solver import SketchyKatyushaState as SketchyKatyushaState
from .solver import SketchySAGA as SketchySAGA
from .solver import SketchySAGAState as SketchySAGAState
from .solver import SketchySGD as SketchySGD
from .solver import SketchySGDState as SketchySGDState
from .solver import SketchySVRG as SketchySVRG
from .solver import SketchySVRGState as SketchySVRGState
from .solver import abstract_cg as abstract_cg
from .solver import abstract_lsqr as abstract_lsqr
from .solver import nystrom_pcg as nystrom_pcg
from .solver import sgmres as sgmres

rand_nystrom_approx = nystrom.rand_nystrom_approx
RandomizedSketching = sketching.RandomizedSketching
GaussianEmbedding = sketching.GaussianEmbedding
SRTT = sketching.SRTT
SparseSignEmbedding = sketching.SparseSignEmbedding
sketch_and_solve = linear_solve.sketch_and_solve
sketch_and_precondition = linear_solve.sketch_and_precondition
SolverState = base.SolverState
LinearOperator = operator.LinearOperator
HessianLinearOperator = operator.HessianLinearOperator
AddLinearOperator = operator.AddLinearOperator
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
tree_size = util.tree_size
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
is_array = util.is_array
sample_indices = util.sample_indices
form_dense_vector = util.form_dense_vector
frozenset = util.frozenset
InputDimError = errors.InputDimError
MatrixNotSquareError = errors.MatrixNotSquareError

__all__ = (
    "rand_nystrom_approx",
    "nystrom_pcg",
    "sketch_and_solve",
    "sketch_and_precondition",
    "sgmres",
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
    "abstract_cg",
    "abstract_lsqr",
    "RandomizedSketching",
    "GaussianEmbedding",
    "SRTT",
    "SparseSignEmbedding",
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
    "tree_size",
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
    "is_array",
    "sample_indices",
    "form_dense_vector",
    "frozenset",
    "InputDimError",
    "MatrixNotSquareError",
)
