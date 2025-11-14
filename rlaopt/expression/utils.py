"""Utility functions for expression module."""

from functools import reduce

import torch

from rlaopt.expression import expr_types


def _create_add(left, right):
    """Create an addition expression with automatic optimizations.

    Optimizations:
    - Flattens nested AddExpressions: (a + b) + c -> AddExpression(a, b, c)
    - Folds constants: const1 + const2 -> single const
    - Filters zeros: expr + 0 -> expr, 0 + expr -> expr

    Args:
        left: Left operand (Expression, float, int, or torch.Tensor).
        right: Right operand (Expression, float, int, or torch.Tensor).

    Returns:
        Expression: Optimized sum (AddExpression, ConstExpression, or single expr).
    """
    # Convert inputs to expressions if needed
    left = _to_expr(left)
    right = _to_expr(right)

    # Collect all terms by flattening nested AddExpressions
    terms = []
    for expr in [left, right]:
        if isinstance(expr, expr_types.add_expr()):
            terms.extend(expr.exprs)
        else:
            terms.append(expr)

    # Separate constants from non-constants
    constants = []
    non_constants = []
    for term in terms:
        if isinstance(term, expr_types.constant()):
            constants.append(term)
        else:
            non_constants.append(term)

    # Fold all constants into one
    if constants:
        total_constants = reduce(torch.add, (c.value for c in constants))
        # Only add back if non-zero (or if it's the only term)
        if (
            not torch.allclose(total_constants, torch.zeros_like(total_constants))
            or len(non_constants) == 0
        ):
            non_constants.append(expr_types.constant()(total_constants))

    # Return based on what's left
    # We do not have to worry about len(non_constants) == 0 because of the check
    # when the constants are folded
    if len(non_constants) == 1:
        return non_constants[0]
    return expr_types.add_expr()(*non_constants)


def _create_product(left, right, matmul: bool):
    """Create a product expression with automatic optimizations.

    Optimizations:
    - Flattens nested ProductExpressions: (a * b) * c -> ProductExpression(a, b, c)
    - Folds scalar constants: 2 * (3 * x) -> 6 * x
    - Distributes constants over sums: const * (a + b) -> const*a + const*b
    - Handles atom scaling: const * atom -> atom._scale(const) if supported

    Does NOT distribute expression * expression products to avoid
    duplicating computation and increasing expression tree size.

    Args:
        left: Left operand (Expression, float, int, or torch.Tensor).
        right: Right operand (Expression, float, int, or torch.Tensor).
        matmul: If True, use matrix multiplication (no optimizations).

    Returns:
        ProductExpression, AddExpression (if distributed), ConstExpression
        (if all constants fold), or AtomExpression (if scaled atom).
    """
    # Convert inputs to expressions if needed
    left = _to_expr(left)
    right = _to_expr(right)

    # Only optimize for elementwise multiplication
    if not matmul:
        # Step 1: Flatten nested ProductExpressions (only elementwise products)
        factors = []
        for expr in [left, right]:
            if isinstance(expr, expr_types.prod_expr()) and not expr.matmul:
                factors.extend(expr.exprs)
            else:
                factors.append(expr)

        # Step 2: Separate scalar constants from other factors
        scalar_constants = []
        other_factors = []
        for factor in factors:
            if isinstance(factor, expr_types.constant()) and factor.value.numel() == 1:
                scalar_constants.append(factor)
            else:
                other_factors.append(factor)

        # Step 3: Fold all scalar constants into one
        folded_const = None
        if scalar_constants:
            product = scalar_constants[0].value.item()
            for const in scalar_constants[1:]:
                product *= const.value.item()
            folded_const = expr_types.constant()(product)

        # Step 4: Apply optimizations with the folded constant
        if folded_const is not None:
            # If only constant(s), return the folded constant
            if len(other_factors) == 0:
                return folded_const

            # If we have exactly one other factor, try special optimizations
            if len(other_factors) == 1:
                other = other_factors[0]

                # Handle atom scaling: const * atom
                from rlaopt.atoms.atom_expression import AtomExpression

                if isinstance(other, AtomExpression):
                    scalar = folded_const.value.item()
                    scaled = other._scale(scalar)
                    if scaled is not NotImplemented:
                        return scaled

                # Distribute: const * (a + b) -> const*a + const*b
                if isinstance(other, expr_types.add_expr()):
                    distributed_terms = [
                        _create_product(folded_const, term, matmul=False)
                        for term in other.exprs
                    ]
                    return expr_types.add_expr()(*distributed_terms)

            # General case: rebuild product with folded constant and other factors
            all_factors = [folded_const] + other_factors
            return expr_types.prod_expr()(*all_factors, matmul=False)

        return expr_types.prod_expr()(*other_factors, matmul=False)

    # No optimization for matmul
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
    if isinstance(val, (float, int, torch.Tensor)) and not isinstance(val, bool):
        return expr_types.constant()(val)
    raise TypeError(f"Cannot convert {type(val).__name__} to Expression")
