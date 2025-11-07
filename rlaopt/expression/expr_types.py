"""Helper module for avoiding circular imports."""


def expression():
    """Return Expression class to avoid circular imports."""
    from rlaopt.expression import expression

    return expression.Expression


def nary_op_expr():
    """Return _NAryOperatorExpression class to avoid circular imports."""
    from rlaopt.expression import _nary_op_expression

    return _nary_op_expression._NAryOpExpression


def unary_op_expr():
    """Return _UnaryOpExpression class to avoid circular imports."""
    from rlaopt.expression import _unary_op_expression

    return _unary_op_expression._UnaryOpExpression


def add_expr():
    """Return AddExpression class to avoid circular imports."""
    from rlaopt.expression import op_expressions

    return op_expressions.AddExpression


def prod_expr():
    """Return ProductExpression class to avoid circular imports."""
    from rlaopt.expression import op_expressions

    return op_expressions.ProductExpression


def constant():
    """Return ConstExpression class to avoid circular imports."""
    from rlaopt.expression import constant

    return constant.ConstExpression


def variable():
    """Return Variable class to avoid circular imports."""
    from rlaopt.expression import variable

    return variable.Variable
