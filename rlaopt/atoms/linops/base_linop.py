import operator
import cvxpy as cp
import torch
from abc import ABC, abstractmethod

from ..atom import AtomExpression

class Linop(AtomExpression, ABC):
    """
    Abstract-base class for linear operators.
    """
    def __init__(self, shape: tuple[int, int] | None):
        super().__init__()
        self._shape = shape

    def __mul__(self, c):
        return _ScaleOperator(c, self)

    def __truediv__(self, c):
        return _ScaleOperator(1/c, self)

    def __rmul__(self, c):
        return _ScaleOperator(c, self)

    def __add__(self, b):
        if isinstance(b, Linop):
            return _BinaryOperator(self, b, operator.add)
        else:
 
    def __sub__(self, b):
        if isinstance(b, Linop):
            return _BinaryOperator(self, b, operator.add)
        else:
            return NotImplemented

    def __neg__(self):
        return -1 * self
    
    def __pow__(self, n):
        return _PowOperator(self, n)

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            S = _SelectorOperator(self.shape[0], key, self)
        if key[1] != slice(None, None, None):
            return NotImplemented
        else:
            return self[key[0]]

    @property
    def shape(self) -> tuple[int, int] | None:
        return self._shape

    def is_smooth(self):
        return True
    
    def is_proxable(self):
        return False
    
    def prox(self, location, prox_scaling):
        raise RuntimeError("Linear functions don't have a prox operator")
    
    def is_subsamplable(self):
        return True
    
    def subsample(self, indices):
        return self[indices, :]

class _ScaleOperator(Linop):
    def __init__(self, c, input_: Linop):
         super().__init__(input_.shape, input_)
        self.register_input(input_)
         self.c = c

    def forward(self):
        input_ = self.get_input()
        if isinstance(input_, Expression):
            value = input_.forward()
        else:
            value = input_
        return self.c * input_

class _BinaryOperator(Linop):
    def __init__(self, left, right, op):
        n, m_L = left.shape
        n, m_R = right.shape
        super().__init__((n, m_L + m_R))
        self.op = op
        self.register_input(left)
        self.register_input(right)
        self.left = left
        self.right = right

    def forward(self):
        left = self.left.forward()
        right = self.right.forward()
    
        return self.op(left, right)

class _SelectorOperator(Linop):
    def __init__(self, key, input_):
        n = len(self.zeros(input_.shape[0])[key])
        super().__init__((n, input_.shape[1]))
        self.register_input(input_)
        self.key = key

    def forward(self):
        return self.get_input().forward()[self.key]
