# Part of the proximal operator implementation is adapted from JAXopt with
# modifications.
# - Original JAXopt proximal operators implementation:
#   https://github.com/google/jaxopt/blob/main/jaxopt/_src/prox.py
# - Original JAXopt projections implementation:
#   https://github.com/google/jaxopt/blob/main/jaxopt/_src/projection.py
#
# Copyright license information:
#
# Copyright 2021 Google LLC
# Modifications copyright 2024 the SketchyOpts authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import jax
import jax.numpy as jnp

from sketchyopts.util import (
    tree_add_scalar_mul,
    tree_l2_norm,
    tree_map,
    tree_scalar_mul,
    tree_vdot,
)


def prox_const(x: Any, hyperparams: Optional[Any] = None, scaling: float = 1.0) -> Any:
    r"""Proximal operator for the constant function.

    For any constant function :math:`f(x) \equiv c` for some constant :math:`c`, and
    any positive scaling factor :math:`\operatorname{scaling}`, the resulting proximal
    operator is the identity operator

    .. math::

      \operatorname{prox}_{\operatorname{scaling} \cdot f}(x)
      = \underset{y}{\argmin}
      \bigg\{
        f(y)
        + \frac{1}{2 \cdot \operatorname{scaling}} \, \lVert y - x \rVert_2^2
      \bigg\}
      = x

    Args:
      x: Pytree input.
      hyperparams: Additional parameters of the proximal operator (for function
        signature consistency). Ignored here.
      scaling: Scaling factor for the proximal operator (for function signature
        consistency). Ignored here.
    Returns:
      Output pytree, with the same structure as ``x``.
    """
    del hyperparams, scaling
    return x


def prox_l1(x: Any, l1reg: Optional[Any] = None, scaling: float = 1.0) -> Any:
    r"""Proximal operator for the :math:`\ell^1` norm. 

  The proximal operator of :math:`f(x) = \lVert x \rVert_1` for :math:`x \in 
  \mathbb{R}^d` is known as the soft-thresholding operator

  .. math::
    
    \begin{aligned}
    \big[
      \operatorname{prox}
      _{\operatorname{scaling} \cdot \operatorname{l1reg} \cdot f}(x)
    \big]_i
    &= \underset{y}{\argmin} 
    \bigg\{
      \operatorname{l1reg} \, \lVert y \rVert_1
      + \frac{1}{2 \cdot \operatorname{scaling}} \, \lVert y - x_i \rVert_2^2
    \bigg\} \\
    &= 
    \begin{cases}
      ~ 0 
        & \lvert x_i \rvert 
        \leqslant \operatorname{scaling} \cdot \operatorname{l1reg} \\
      ~ x_i 
      - \operatorname{scaling} \cdot \operatorname{l1reg} \cdot 
      \operatorname{sign}(x_i) 
        & \lvert x_i \rvert 
        > \operatorname{scaling} \cdot \operatorname{l1reg} \\
    \end{cases} \\
    \end{aligned}

  In lasso regression, the function is used as :math:`\ell^1`-regularization.  

  Args:
    x: Pytree input.
    l1reg: Regularization strength. It could either be scalar (same strength across 
      tree leaves) or a pytree with the same structure as ``x`` (leaf-wise strength).
    scaling: Scaling factor for the proximal operator. 

  Returns:
    Output pytree, with the same structure as ``x``.
  """
    if l1reg is None:
        l1reg = 1.0

    if jnp.isscalar(l1reg):
        l1reg = tree_map(lambda y: l1reg * jnp.ones_like(y), x)

    def fun(u, v):
        return jnp.sign(u) * jax.nn.relu(jnp.abs(u) - v * scaling)

    return tree_map(fun, x, l1reg)


def prox_nonnegative_l1(
    x: Any, l1reg: Optional[float] = None, scaling: float = 1.0
) -> Any:
    r"""Proximal operator for the :math:`\ell^1` norm on the nonnegative orthant. 

  The proximal operator of :math:`f(x) = \lVert x \rVert_1 + 
  \mathbb{1}_{\mathbb{R}_{+}^{d}}(x)` for :math:`x \in \mathbb{R}^d` is given by

  .. math::

    \begin{aligned}
    \big[
      \operatorname{prox}
      _{\operatorname{scaling} \cdot \operatorname{l1reg} \cdot f}(x)
    \big]_i
    &= \underset{y}{\argmin} 
    \bigg\{
      \operatorname{l1reg} \, 
      \big(\lVert y \rVert_1 + \mathbb{1}_{\mathbb{R}_{+}}(y)\big)
      + \frac{1}{2 \cdot \operatorname{scaling}} \, \lVert y - x_i \rVert_2^2
    \bigg\} \\
    &= \underset{y \geqslant 0}{\argmin} 
    \bigg\{
      \operatorname{l1reg} \, \lVert y \rVert_1 
      + \frac{1}{2 \cdot \operatorname{scaling}} \, \lVert y - x_i \rVert_2^2
    \bigg\} \\
    &= 
    \begin{cases}
      ~ 0 
        & x_i \leqslant \operatorname{scaling} \cdot \operatorname{l1reg} \\
      ~ x_i - \operatorname{scaling} \cdot \operatorname{l1reg} 
        & x_i > \operatorname{scaling} \cdot \operatorname{l1reg} \\
    \end{cases} \\
    \end{aligned}

  where :math:`\mathbb{1}_{\mathbb{R}_{+}^{d}}` is the indicator function of 
  nonnegative orthant :math:`\mathbb{R}_{+}^{d} \colonequals 
  \{x \in \mathbb{R}^d \mid x \succcurlyeq 0\}`

  .. math::

    \mathbb{1}_{\mathbb{R}_{+}^{d}}(a)
    = 
    \begin{cases}
      ~ 0 
        & a_i \geqslant 0 ~\text{for}~i = 1, \cdots, d \\
      ~ \infty
        & \text{otherwise}
    \end{cases}

  Args:
    x: Pytree input.
    l1reg: Regularization strength. It could either be scalar (same strength across 
      tree leaves) or a pytree with the same structure as ``x`` (leaf-wise strength).
    scaling: Scaling factor for the proximal operator. 

  Returns:
    Output pytree, with the same structure as ``x``.
  """
    if l1reg is None:
        l1reg = 1.0

    pytree = tree_map(lambda y: y - l1reg * scaling, x)

    return tree_map(jax.nn.relu, pytree)


def prox_elastic_net(
    x: Any, hyperparams: Optional[tuple[Any, Any]] = None, scaling: float = 1.0
) -> Any:
    r"""Proximal operator for the elastic net.

  The proximal operator of 

  .. math::

    f_{\operatorname{hyperparams}}(x) 
    = \operatorname{hyperparams}_0 \, \lVert x \rVert_1
    + \frac{\operatorname{hyperparams}_1}{2} \, \lVert x \rVert_2^2

  for :math:`x \in \mathbb{R}^d` is given by soft-thresholding followed by 
  multiplicative shrinkage

  .. math::

    \begin{aligned}
    \big[
      \operatorname{prox}
      _{\operatorname{scaling} \cdot f_{\operatorname{hyperparams}}}(x)
    \big]_i
    &= \bigg[\underset{y}{\argmin} 
    \bigg\{
      f_{\operatorname{hyperparams}}(y)
      + \frac{1}{2 \cdot \operatorname{scaling}} \, \lVert y - x \rVert_2^2
    \bigg\}\bigg]_i \\
    &= \frac{
      \psi_{\operatorname{scaling} \cdot \operatorname{hyperparams}_0}(x_i)
    }{1 + \operatorname{scaling} \cdot \operatorname{hyperparams}_1} \\
    \end{aligned}

  where :math:`\psi_{\operatorname{scaling} \cdot \operatorname{hyperparams}_0}` is the 
  soft-thresholding operator parameterized by :math:`\operatorname{scaling} \cdot 
  \operatorname{hyperparams}_0`
  
  .. math::
  
    \psi(x_i) = 
    \begin{cases}
      ~ 0 
        & \lvert x_i \rvert 
        \leqslant \operatorname{scaling} \cdot \operatorname{hyperparams}_0 \\
      ~ x_i 
      - \operatorname{scaling} \cdot \operatorname{hyperparams}_0 \cdot 
      \operatorname{sign}(x_i) 
        & \lvert x_i \rvert 
        > \operatorname{scaling} \cdot \operatorname{hyperparams}_0 \\
    \end{cases}

  Args:
    x: Pytree input.
    hyperparams: A tuple, where both ``hyperparams[0]`` and ``hyperparams[1]`` can be 
      either floats or pytrees with the same structure as ``x``.
    scaling: Scaling factor for the proximal operator. 

  Returns:
    Output pytree, with the same structure as ``x``.
  """
    if hyperparams is None:
        hyperparams = (1.0, 1.0)

    lam = (
        tree_map(lambda y: hyperparams[0] * jnp.ones_like(y), x)
        if jnp.isscalar(hyperparams[0])
        else hyperparams[0]
    )
    gam = (
        tree_map(lambda y: hyperparams[1] * jnp.ones_like(y), x)
        if jnp.isscalar(hyperparams[1])
        else hyperparams[1]
    )

    def prox_l1(u, lambd):
        return jnp.sign(u) * jax.nn.relu(jnp.abs(u) - lambd)

    def fun(u, lambd, gamma):
        return prox_l1(u, scaling * lambd) / (1.0 + scaling * gamma)

    return tree_map(fun, x, lam, gam)


def prox_l2(x: Any, l2reg: Optional[float] = 1.0, scaling=1.0) -> Any:
    r"""Proximal operator for the :math:`\ell^2` norm. 
  
  The proximal operator of :math:`f(x) = \lVert x \rVert_2` for :math:`x \in 
  \mathbb{R}^d` is known as the block soft-thresholding operator

  .. math::

    \begin{aligned}
    \operatorname{prox}
    _{\operatorname{scaling} \cdot \operatorname{l2reg} \cdot f}(x)
    &= \underset{y}{\argmin} 
    \bigg\{
      \operatorname{l2reg} \, \lVert y \rVert_2
      + \frac{1}{2 \cdot \operatorname{scaling}} \, \lVert y - x \rVert_2^2
    \bigg\} \\
    &= 
    \begin{cases}
      ~ 0 
        & \lVert x \rVert_2
        \leqslant \operatorname{scaling} \cdot \operatorname{l2reg} \\
      ~ \bigg(1 - \dfrac
      {\operatorname{scaling} \cdot \operatorname{l2reg}}
      {\lVert x \rVert_2}\bigg) \, x
        & \lVert x \rVert_2
        > \operatorname{scaling} \cdot \operatorname{l2reg} \\
    \end{cases} \\
    \end{aligned}
  
  In group lasso, the function (also known as :math:`\ell^1/\ell^2` norm) takes the form

  .. math::

    f(x) = \sum_{g \in \mathcal{G}} \lVert x_{\mathcal{I}_g} \rVert_2

  where :math:`\mathcal{G}` is a partition of :math:`[d]` and :math:`\mathcal{I}_g` is 
  the index set belonging to the :math:`g`:sup:`th` group. The corresponding proximal 
  operator in this case is given by

  .. math::

    \big[
      \operatorname{prox}
      _{\operatorname{scaling} \cdot \operatorname{l2reg} \cdot f}(x)
    \big]_{i}
    = 
    \begin{cases}
      ~ 0 
        & \lVert x_{\mathcal{I}_g} \rVert_2
        \leqslant \operatorname{scaling} \cdot \operatorname{l2reg} \\
      ~ \bigg(1 - \dfrac
      {\operatorname{scaling} \cdot \operatorname{l2reg}}
      {\lVert x_{\mathcal{I}_g} \rVert_2}\bigg) \, x_{i}
        & \lVert x_{\mathcal{I}_g} \rVert_2
        > \operatorname{scaling} \cdot \operatorname{l2reg} \\
    \end{cases} \\

  for all :math:`i \in \mathcal{I}_g` and :math:`g \in \mathcal{G}`. 

  .. note:: 
    The proximal operator is implemented for the :math:`\ell^2` norm (*i.e.* single 
    group case). To use it for group lasso, one can use :func:`jax.vmap` or 
    :func:`jax.tree.map` to apply the proximal mapping to groups as shown in the 
    following example. 

  Example:
    .. highlight:: python
    .. code-block:: python

      import jax
      from sketchyopts.prox import prox_l2

      # array with 2 or more dimensions and first axis across groups
      def prox_group_lasso_array(x, l2reg=1.0, scaling=1.0): 
        # collapse each group into a single dimension
        groups = jax.lax.collapse(x, 1)
        # apply the operator to each group
        result = jax.vmap(prox_l2, (0, None, None))(groups, l2reg, scaling)
        # make result the original shape
        return result.reshape(x.shape)

      # pytree with groups as leaves
      def prox_group_lasso_pytree(x, l2reg=1.0, scaling=1.0): 
        # define partial function that only depends on the pytree
        partial_fun = lambda u: prox_l2(u, l2reg, scaling)
        # apply the operator to each group
        result = jax.tree.map(partial_fun, x)
        return result

  Args:
    x: Pytree input.
    l2reg: Regularization strength.
    scaling: Scaling factor for the proximal operator.

  Returns:
    Output pytree, with the same structure as ``x``.
  """
    if l2reg is None:
        l2reg = 1.0

    l2_norm = tree_l2_norm(x)
    factor = 1.0 - l2reg * scaling / l2_norm
    factor = jnp.where(factor >= 0, factor, 0)

    return tree_scalar_mul(factor, x)


def prox_l2_squared(x: Any, l2reg: Optional[float] = 1.0, scaling=1.0) -> Any:
    r"""Proximal operator for the squared :math:`\ell^2` norm. 

  The proximal operator of :math:`f(x) = (1/2) \, \lVert x \rVert_2^2` for :math:`x \in 
  \mathbb{R}^d` is given by

  .. math::

    \begin{aligned}
    \operatorname{prox}
    _{\operatorname{scaling} \cdot \operatorname{l2reg} \cdot f}(x)
    &= \underset{y}{\argmin} 
    \bigg\{
      \frac{\operatorname{l2reg}}{2} \, \lVert y \rVert_2^2
      + \frac{1}{2 \cdot \operatorname{scaling}} \, \lVert y - x \rVert_2^2
    \bigg\} \\
    &= \frac{1}{1 + \operatorname{scaling} \cdot \operatorname{l2reg}} \, x \\
    \end{aligned}

  In ridge regression, the function is used as :math:`\ell^2`-regularization.  

  Args:
    x: Pytree input.
    l2reg: Regularization strength.
    scaling: Scaling factor for the proximal operator.

  Returns:
    Output pytree, with the same structure as ``x``.
  """
    if l2reg is None:
        l2reg = 1.0

    factor = 1.0 / (1.0 + scaling * l2reg)

    return tree_scalar_mul(factor, x)


def prox_nonnegative_l2_squared(
    x: Any, l2reg: Optional[float] = 1.0, scaling: float = 1.0
) -> Any:
    r"""Proximal operator for the squared :math:`\ell^2` norm on the nonnegative 
  orthant.

  The proximal operator of :math:`f(x) = (1/2) \, \lVert x \rVert_2^2 + 
  \mathbb{1}_{\mathbb{R}_{+}^{d}}(x)` for :math:`x \in \mathbb{R}^d` is given by

  .. math::

    \begin{aligned}
    \operatorname{prox}
    _{\operatorname{scaling} \cdot \operatorname{l2reg} \cdot f}(x)
    &= \underset{y}{\argmin} 
    \bigg\{
      \operatorname{l2reg} \, 
      \bigg(
        \frac{1}{2} \, \lVert y \rVert_2^2 
        + \mathbb{1}_{\mathbb{R}_{+}^{d}}(y)
      \bigg)
      + \frac{1}{2 \cdot \operatorname{scaling}} \, \lVert y - x \rVert_2^2
    \bigg\} \\
    &= \underset{y \succcurlyeq 0}{\argmin} 
    \bigg\{
      \frac{\operatorname{l2reg}}{2} \, \lVert y \rVert_2^2
      + \frac{1}{2 \cdot \operatorname{scaling}} \, \lVert y - x \rVert_2^2
    \bigg\} \\
    &= \bigg(
      \frac{1}{1 + \operatorname{scaling} \cdot \operatorname{l2reg}} \, x
      \bigg)_{+} \\
    \end{aligned}

  where :math:`\mathbb{1}_{\mathbb{R}_{+}^{d}}` is the indicator function of nonnegative 
  orthant :math:`\mathbb{R}_{+}^{d} \colonequals \{x \in \mathbb{R}^d \mid x 
  \succcurlyeq 0\}`

  .. math::

    \mathbb{1}_{\mathbb{R}_{+}^{d}}(a)
    = 
    \begin{cases}
      ~ 0 
        & a_i \geqslant 0 ~\text{for}~i = 1, \cdots, d \\
      ~ \infty
        & \text{otherwise}
    \end{cases}

  and :math:`(a)_{+} \colonequals \big[\max \{a_i, 0\}\big]_{i = 1, \cdots, d}` takes 
  element-wise positive part of :math:`a`. 

  Args:
    x: Pytree input.
    l2reg: Regularization strength.
    scaling: Scaling factor for the proximal operator.

  Returns:
    Output pytree, with the same structure as ``x``.
  """
    if l2reg is None:
        l2reg = 1.0

    pytree = tree_scalar_mul(1.0 / (1.0 + l2reg * scaling), x)

    return tree_map(jax.nn.relu, pytree)


def prox_nonnegative(
    x: Any, hyperparams: Optional[Any] = None, scaling: float = 1.0
) -> Any:
    r"""Proximal operator for indicator of the nonnegative orthant. 
  
  The proximal operator of indicator :math:`f(x) = \mathbb{1}_{\mathbb{R}_{+}^{d}}(x)` 
  for :math:`x \in \mathbb{R}^d` is the projection onto the nonnegative orthant. Here 
  :math:`\mathbb{1}_{\mathbb{R}_{+}^{d}}` is the indicator function of nonnegative 
  orthant :math:`\mathbb{R}_{+}^{d} \colonequals \{x \in \mathbb{R}^d \mid x 
  \succcurlyeq 0\}`

  .. math::

    \mathbb{1}_{\mathbb{R}_{+}^{d}}(a)
    = 
    \begin{cases}
      ~ 0 
        & a_i \geqslant 0 ~\text{for}~i = 1, \cdots, d \\
      ~ \infty
        & \text{otherwise}
    \end{cases}

  The proximal operator is given by 

  .. math::

    \begin{aligned}
    \operatorname{prox}_{\operatorname{scaling} \cdot f}(x)
    &= \underset{y}{\argmin} 
    \bigg\{
      \mathbb{1}_{\mathbb{R}_{+}}(y)
      + \frac{1}{2 \cdot \operatorname{scaling}} \, \lVert y - x \rVert_2^2
    \bigg\} \\
    &= \underset{y \succcurlyeq 0}{\argmin} 
    \bigg\{
      \lVert y - x \rVert_2^2
    \bigg\} \\
    &= (x)_{+} \\
    \end{aligned}

  where :math:`(a)_{+} \colonequals \big[\max \{a_i, 0\}\big]_{i = 1, \cdots, d}` takes 
  element-wise positive part of :math:`a`. 

  Args:
    x: Pytree input.
    hyperparams: Additional parameters of the proximal operator (for function 
      signature consistency). Ignored here.
    scaling: Scaling factor for the proximal operator (for function signature
      consistency). Ignored here. 
  Returns:
    Output pytree, with the same structure as ``x``.
  """
    del hyperparams, scaling
    return tree_map(jax.nn.relu, x)


def _clip_safe(x, lower, upper):
    return jnp.clip(jnp.asarray(x), lower, upper)


def prox_box(x: Any, hyperparams: tuple, scaling: float = 1.0) -> Any:
    r"""Proximal operator for indicator of a box (high-dimensional closed interval). 

  The proximal operator of indicator :math:`f(x) = \mathbb{1}_{\mathcal{C}}(x)` for set
  
  .. math::

    \mathcal{C}
    \colonequals
    \{x \in \mathbb{R}^d \mid 
      \operatorname{hyperparams}_0
        \preccurlyeq x \preccurlyeq
      \operatorname{hyperparams}_1
    \}

  is the projection onto the set :math:`\mathcal{C}`. Here :math:`\mathbb{1}_
  {\mathcal{C}}` is the indicator function of the box :math:`\mathcal{C}`

  .. math::

    \mathbb{1}_{\mathcal{C}}(a)
    = 
    \begin{cases}
      ~ 0 
        & \big[\operatorname{hyperparams}_0\big]_i 
          \leqslant a_i \leqslant 
          \big[\operatorname{hyperparams}_1\big]_i 
        ~\text{for}~i = 1, \cdots, d \\
      ~ \infty
        & \text{otherwise}
    \end{cases}

  The proximal operator is given by 

  .. math::

    \begin{aligned}
    \operatorname{prox}_{\operatorname{scaling} \cdot f}(x)
    &= \underset{y}{\argmin} 
    \bigg\{
      \mathbb{1}_{\mathcal{C}}(y)
      + \frac{1}{2 \cdot \operatorname{scaling}} \, \lVert y - x \rVert_2^2
    \bigg\} \\
    &= \underset{y \in \mathcal{C}}{\argmin} 
    \bigg\{
      \lVert y - x \rVert_2^2
    \bigg\} \\
    &= \max\big\{
      \operatorname{hyperparams}_0, \,
      \min\{x, \, \operatorname{hyperparams}_1\}
    \big\} \\
    \end{aligned}

  where both the minimum and maximum are element-wise operations. Intuitively, the 
  proximal operator clamps :math:`x` into the interval. 

  Args:
    x: Pytree input.
    hyperparams: A tuple ``(lower, upper)`` specifying lower and upper bounds of the 
      box. Both bounds can be either scalar values or pytrees of the same structure as 
      ``x``.
    scaling: Scaling factor for the proximal operator (for function signature
      consistency). Ignored here. 
  Returns:
    Output pytree, with the same structure as ``x``.
  """
    del scaling
    lower, upper = hyperparams

    if jnp.isscalar(lower):
        lower = tree_map(lambda y: lower * jnp.ones_like(y), x)

    if jnp.isscalar(upper):
        upper = tree_map(lambda y: upper * jnp.ones_like(y), x)

    return tree_map(_clip_safe, x, lower, upper)


def prox_hyperplane(x: Any, hyperparams: tuple, scaling: float = 1.0) -> Any:
    r"""Proximal operator for indicator of a hyperplane. 

  The proximal operator of indicator :math:`f(x) = \mathbb{1}_{\mathcal{H}}(x)` for set
  
  .. math::

    \mathcal{H}
    \colonequals
    \{x \in \mathbb{R}^d \mid 
      \operatorname{hyperparams}_0^{\mathsf{T}} x 
      = \operatorname{hyperparams}_1
    \}

  is the projection onto the set :math:`\mathcal{H}`. Here :math:`\mathbb{1}_
  {\mathcal{H}}` is the indicator function of the hyperplane :math:`\mathcal{H}`

  .. math::

    \mathbb{1}_{\mathcal{H}}(a)
    = 
    \begin{cases}
      ~ 0 
        & \operatorname{hyperparams}_0^{\mathsf{T}} a
        = \operatorname{hyperparams}_1 \\
      ~ \infty
        & \text{otherwise}
    \end{cases}

  The proximal operator is given by 

  .. math::

    \begin{aligned}
    \operatorname{prox}_{\operatorname{scaling} \cdot f}(x)
    &= \underset{y}{\argmin} 
    \bigg\{
      \mathbb{1}_{\mathcal{H}}(y)
      + \frac{1}{2 \cdot \operatorname{scaling}} \, \lVert y - x \rVert_2^2
    \bigg\} \\
    &= \underset{y \in \mathcal{H}}{\argmin} 
    \bigg\{
      \lVert y - x \rVert_2^2
    \bigg\} \\
    &= x 
    - \frac
    {
      \operatorname{hyperparams}_0^{\mathsf{T}} x 
      - \operatorname{hyperparams}_1
    }{
      \lVert \operatorname{hyperparams}_0 \rVert_2^2
    } \, \operatorname{hyperparams}_0 \\
    \end{aligned}

  Args:
    x: Pytree input.
    hyperparams: A tuple ``(a, b)`` specifying both the normal vector and the offset of 
      the hyperplane. The normal vector ``a`` is a pytree with the same structure as 
      ``x`` and the offset ``b`` is a scalar. 
    scaling: Scaling factor for the proximal operator (for function signature
      consistency). Ignored here. 
  Returns:
    Output pytree, with the same structure as ``x``.
  """
    del scaling
    a, b = hyperparams
    scale = (tree_vdot(a, x) - b) / tree_vdot(a, a)

    return tree_add_scalar_mul(x, -scale, a)


def prox_halfspace(x: Any, hyperparams: tuple, scaling: float = 1.0) -> Any:
    r"""Proximal operator for indicator of a halfspace. 

  The proximal operator of indicator :math:`f(x) = \mathbb{1}_{\mathcal{H}}(x)` for set
  
  .. math::

    \mathcal{H}
    \colonequals
    \{x \in \mathbb{R}^d \mid 
      \operatorname{hyperparams}_0^{\mathsf{T}} x 
      \leqslant \operatorname{hyperparams}_1
    \}

  is the projection onto the set :math:`\mathcal{H}`. Here :math:`\mathbb{1}_
  {\mathcal{H}}` is the indicator function of the halfspace :math:`\mathcal{H}`

  .. math::

    \mathbb{1}_{\mathcal{H}}(a)
    = 
    \begin{cases}
      ~ 0 
        & \operatorname{hyperparams}_0^{\mathsf{T}} a
        \leqslant \operatorname{hyperparams}_1 \\
      ~ \infty
        & \text{otherwise}
    \end{cases}

  The proximal operator is given by 

  .. math::

    \begin{aligned}
    \operatorname{prox}_{\operatorname{scaling} \cdot f}(x)
    &= \underset{y}{\argmin} 
    \bigg\{
      \mathbb{1}_{\mathcal{H}}(y)
      + \frac{1}{2 \cdot \operatorname{scaling}} \, \lVert y - x \rVert_2^2
    \bigg\} \\
    &= \underset{y \in \mathcal{H}}{\argmin} 
    \bigg\{
      \lVert y - x \rVert_2^2
    \bigg\} \\
    &= x 
    - \frac
    {
      (\operatorname{hyperparams}_0^{\mathsf{T}} x 
      - \operatorname{hyperparams}_1)_{+}
    }{
      \lVert \operatorname{hyperparams}_0 \rVert_2^2
    } \, \operatorname{hyperparams}_0 \\
    \end{aligned}

  where :math:`(a)_{+} \colonequals \max \{a, 0\}` takes the positive part of :math:`a`. 

  Args:
    x: Pytree input.
    hyperparams: A tuple ``(a, b)`` specifying both the normal vector and the offset of 
      the hyperplane that defines the halfspace. The normal vector ``a`` is a pytree 
      with the same structure as ``x`` and the offset ``b`` is a scalar. 
    scaling: Scaling factor for the proximal operator (for function signature
      consistency). Ignored here. 
  Returns:
    Output pytree, with the same structure as ``x``.
  """
    del scaling
    a, b = hyperparams
    scale = jax.nn.relu(tree_vdot(a, x) - b) / tree_vdot(a, a)

    return tree_add_scalar_mul(x, -scale, a)
