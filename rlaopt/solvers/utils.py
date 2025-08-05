from ..expression.expression import Expression, AddExpression
from ..params.params import Params

def split_objective(obj: Expression | AddExpression):
     # Split the objective if we have an AddExpression
     if isinstance(obj, AddExpression):
          f, r = obj.operator_split()
          if r:
             # Raise error if r is not proxable
             if not r.is_proxable:
                  raise RuntimeError("Input obj has non-proxable non-smooth term, which is not supported")
             # If proxable use r's prox operator
             else:
                  if obj._num_non_smooth_exprs == 1:
                       def prox(params: Params, eta: float) -> Params:
                            return params.fmap(lambda v: r.prox(v, eta))
                  else:
                       def prox(params: Params, eta: float) -> Params:
                            return Params(r.prox(params.value, eta))
                       
          # When r is none, obj is smooth, so prox is just the identity
          else:
               prox = id_prox  
     
     # If obj is just an expression, then assume obj is smooth.  
     # Prox is just the identity map as there is no non-smooth term.
     elif isinstance(obj, Expression):
          f = obj
          prox = id_prox
     
     else: 
        raise TypeError(f"obj must be of type Expression|AddExpression but got {type(obj)}")
     return f, prox


def id_prox(params: Params, prox_scaling: float) -> Params:
     return params       