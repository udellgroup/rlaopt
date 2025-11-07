"""Utility functions for expression module."""

import torch

from rlaopt.expression import expr_types


def _create_product(left, right, matmul: bool):
    """Create a product expression with automatic distribution.

    Distributes constants over sums to enable better splitting:
    - const * (a + b + c) -> const*a + const*b + const*c
    - (a + b + c) * const -> a*const + b*const + c*const

    Condenses products of constants:
    - const1 * const2 -> single const with combined value

    Handles atom scaling:
    - const * atom -> atom._scale(const.value) if atom supports it
    - atom * const -> atom._scale(const.value) if atom supports it

    Does NOT distribute expression * expression products to avoid
    duplicating computation and increasing expression tree size.

    Args:
        left: Left operand (Expression, float, int, or torch.Tensor).
        right: Right operand (Expression, float, int, or torch.Tensor).
        matmul: If True, use matrix multiplication (no distribution).

    Returns:
        ProductExpression, AddExpression (if distributed), ConstExpression
        (if both constants), or AtomExpression (if scaled atom).
    """
    # Convert inputs to expressions if needed
    left = _to_expr(left)
    right = _to_expr(right)

    # Only optimize for elementwise multiplication
    if not matmul:
        # Check if we have constants
        left_is_const = isinstance(left, expr_types.constant())
        right_is_const = isinstance(right, expr_types.constant())

        # Condense: const * const -> single const
        if left_is_const and right_is_const:
            combined_value = left.value * right.value
            return expr_types.constant()(combined_value)

        # Handle atom scaling: const * atom or atom * const
        # Import here to avoid circular imports
        from rlaopt.atoms.atom_expression import AtomExpression

        # const * atom
        if left_is_const and isinstance(right, AtomExpression):
            if left.value.numel() == 1:
                scalar = left.value.item()
                try:
                    return right._scale(scalar)
                except NotImplementedError:
                    pass

        # atom * const
        if right_is_const and isinstance(left, AtomExpression):
            if right.value.numel() == 1:
                scalar = right.value.item()
                try:
                    return left._scale(scalar)
                except NotImplementedError:
                    pass

        # Distribute: const * (a + b) -> const*a + const*b
        if left_is_const and isinstance(right, expr_types.add_expr()):
            distributed_terms = [
                _create_product(left, term, matmul=False) for term in right.exprs
            ]
            return expr_types.add_expr()(*distributed_terms)

        # Distribute: (a + b) * const -> a*const + b*const
        if right_is_const and isinstance(left, expr_types.add_expr()):
            distributed_terms = [
                _create_product(term, right, matmul=False) for term in left.exprs
            ]
            return expr_types.add_expr()(*distributed_terms)

    # No distribution needed
    return expr_types.prod_expr()(left, right, matmul=matmul)


def _to_expr(val):
    """Convert a value to an Expression if it isn't already.

    Args:
        val: Value to convert. Can be an Expression, float, int, or torch.Tensor.

    Returns:
        Expression: The converted expression.

    Raises:
        TypeError: If val cannot be converted to an Expression.

    """
    if isinstance(val, expr_types.expression()):
        return val
    if isinstance(val, (float, int, torch.Tensor)):
        return expr_types.constant()(val)
    raise TypeError(f"Cannot convert {type(val)} to Expression")
