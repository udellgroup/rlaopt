"""SapphireSplit class for representing finite-sum composite objective functions."""

from functools import partial
from typing import Callable

import torch
from linops import LinearOperator

from rlaopt.atoms import Atom
from rlaopt.atoms.linear_model.linear_model import LinearModel
from rlaopt.data import DataLoader
from rlaopt.expression import AddExpression, Expression
from rlaopt.ext_tensordict import TensorDict


class SapphireSplit:
    """Splitting class for SAPPHIRE optimization algorithm.

    Splits objective into a sum of three terms:

        loss + f + r

    where:
        1. loss: Linear model loss
        2. f: Smooth regularizer
        3. r: Non-smooth regularizer

    Args:
        expr (Expression): An expression object representing the composite function.

    Raises:
        ValueError: If the expression cannot be split into the required form.

    Attributes:
        model (LinearModel): The linear model term from the objective.
        f (Expression | None): The smooth regularizer term.
        r (Atom | None): The non-smooth regularizer term.
    """

    def __init__(self, expr: Expression):
        """Initialize SAPPHIRE split from objective expression.

        Args:
            expr (Expression): An expression object representing the composite function.
        """
        model, f, r = self._attempt_split(expr)

        self._model = model
        self._f = f
        self._r = r

    def _attempt_split(
        self, expr: AddExpression
    ) -> tuple[LinearModel, AddExpression | None, Atom | None]:
        """Attempts to split the objective as: Linear Model + Smooth + Non-smooth.

        Args:
            expr (AddExpression): The expression to split.

        Returns:
            tuple[LinearModel, AddExpression | None, Atom | None]: A tuple containing:
                - LinearModel: The linear model term
                - AddExpression | None: The smooth regularizer term (if present)
                - Atom | None: The non-smooth regularizer term (if present)

        Raises:
            ValueError: If expression cannot be split for SAPPHIRE. Specifically:
                - If more than one non-smooth expression is present
                - If no LinearModel term is present
                - If multiple LinearModel terms are present
                - If smooth expressions depend on variables other than beta
                - If non-smooth expression depends on variables other than beta
        """
        smooth_part = expr.get_smooth_part(return_mode="list")
        non_smooth_exprs = expr.get_non_smooth_exprs()

        # Validate non-smooth expressions
        if non_smooth_exprs and len(non_smooth_exprs) > 1:
            raise ValueError(
                f"Regularizer can only consist of one non-smooth expression "
                f"but received {len(non_smooth_exprs)}."
            )

        non_smooth_expr = non_smooth_exprs[0] if non_smooth_exprs else None

        # Separate smooth expressions into LinearModel and other expressions
        smooth_exprs = []
        linear_model_exprs = []
        for sub_expr in smooth_part:
            if isinstance(sub_expr, LinearModel):
                linear_model_exprs.append(sub_expr)
            else:
                smooth_exprs.append(sub_expr)

        # Validate LinearModel presence
        if not linear_model_exprs:
            raise ValueError(
                "Objective missing term of type LinearModel. "
                "Objective must contain one and only one LinearModel term."
            )

        if len(linear_model_exprs) > 1:
            raise ValueError(
                "Smooth part of the objective can only have one expression of type LinearModel."
            )

        linear_model = linear_model_exprs[0]

        # Validate smooth expressions depend only on beta
        if smooth_exprs:
            smooth_var_names = set()
            for expr in smooth_exprs:
                smooth_var_names.update(expr.get_variable_names())

            if smooth_var_names != {"beta"}:
                raise ValueError(
                    "Smooth expression depends upon variables other than beta. "
                    "SAPPHIRE solver only supports smooth expressions that depend on beta."
                )

        # Validate non-smooth expression depends only on beta (only if present)
        if non_smooth_expr is not None:
            non_smooth_var_names = set(non_smooth_expr.get_variable_names())
            if non_smooth_var_names != {"beta"}:
                raise ValueError(
                    "Non-smooth regularizer depends upon variables other than beta. "
                    "SAPPHIRE solver only supports non-smooth expressions that depend on beta."
                )

        # Prepare return values
        smooth_expr = AddExpression(*smooth_exprs) if smooth_exprs else None

        return linear_model, smooth_expr, non_smooth_expr

    @property
    def f(self) -> Expression | None:
        """Returns smooth regularizer.

        Returns:
            Expression | None: The smooth regularizer term if present, None otherwise.
        """
        return self._f

    @property
    def r(self) -> Atom | None:
        """Returns non-smooth regularizer.

        Returns:
            Atom | None: The non-smooth regularizer term if present, None otherwise.
        """
        return self._r

    @property
    def model(self) -> LinearModel:
        """Returns linear model term.

        Returns:
            LinearModel: The linear model term from the objective.
        """
        return self._model

    @property
    def loader(self) -> DataLoader:
        """Convenience property for accessing data loader.

        Returns:
            DataLoader: The data loader from the linear model term.
        """
        return self._model.dataloader

    @property
    def num_samples(self) -> int:
        """Convenience property for accessing number of samples.

        Returns the number of samples from the linear model term's dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self._model.dataloader.dataset.num_samples

    @property
    def variable_values(self) -> TensorDict:
        """Returns the variable values associated with the linear model.

        Returns:
            TensorDict: The current variable values (weights) of the linear model.
        """
        return self._model.variable_values

    def evaluate(self, beta_value: TensorDict) -> torch.Tensor:
        """Evaluate the composite objective function at the given variables.

        Computes the full objective: loss + f + r.

        Args:
            beta_value (TensorDict): Linear model weights.

        Returns:
            torch.Tensor: The scalar value of the objective function at beta_value.
        """
        val = self.model.evaluate(beta_value)
        if self.f:
            val += self._f.evaluate(beta_value)
        elif self.r:
            val += self._r.evaluate(beta_value)

        return val

    def loss(
        self, beta_value: TensorDict, X_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates smooth part of the loss.

        Computes the sum of the linear model loss and smooth regularizer: loss + f.

        Args:
            beta_value (TensorDict): Linear model weights.
            X_batch (torch.Tensor): Batch of feature data.
            y_batch (torch.Tensor): Batch of target labels.

        Returns:
            torch.Tensor: The scalar value of the smooth loss.
        """
        val = self._model.loss(beta_value, X_batch, y_batch)
        if self._f:
            val += self._f.evaluate(beta_value)
        return val

    def batch_loss_grad(
        self, beta_value: TensorDict, X_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> TensorDict:
        """Computes gradient of the smooth loss on a batch.

        Args:
            beta_value (TensorDict): Linear model weights.
            X_batch (torch.Tensor): Batch of feature data.
            y_batch (torch.Tensor): Batch of target labels.

        Returns:
            TensorDict: Gradient of the smooth loss with respect to beta.
        """
        return torch.func.grad(self.loss)(beta_value, X_batch, y_batch)

    def gradient(self, beta_value: TensorDict) -> TensorDict:
        """Computes gradient of the full smooth loss.

        Args:
            beta_value (TensorDict): Linear model weights.

        Returns:
            TensorDict: Gradient of the full smooth loss with respect to beta.
        """
        return torch.func.grad(self.loss)(beta_value, None, None)

    def prox(self, beta_value: TensorDict, eta: float) -> TensorDict:
        """Apply the proximal operator of r with step size eta to the variables.

        Args:
            beta_value (TensorDict): A dictionary of variables.
            eta (float): Step size or scaling factor for the proximal operator.

        Returns:
            TensorDict: Updated variables after applying the proximal operator.
        """
        if self._r:
            beta_value_values_update = self.r.prox(beta_value, eta)
            beta_value.update(beta_value_values_update)
        return beta_value

    def get_subsamp_hessian_linop(
        self,
        beta_value: TensorDict,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        device: torch.device,
    ):
        """Construct subsampled Hessian linear operator.

        Args:
            beta_value (TensorDict): Linear model weights at which to evaluate the Hessian.
            X_batch (torch.Tensor): Batch of feature data for subsampling.
            y_batch (torch.Tensor): Batch of target labels for subsampling.
            device (torch.device): Device on which the model parameters live.

        Returns:
            _SubampHessianLinOp: A linear operator representing the subsampled Hessian.
        """
        return _SubampHessianLinOp(
            self._model.loss, beta_value, X_batch, y_batch, device
        )


class _SubampHessianLinOp(LinearOperator):
    """Subsampled Hessian linear operator class.

    Implements a linear operator interface for computing Hessian-vector products
    using a subsampled batch of data. Uses forward-over-reverse automatic
    differentiation for efficient computation.

    Args:
        loss (Callable): Loss function from which subsampled Hessian is constructed.
            Should accept a TensorDict and return a scalar tensor.
        variable_values (TensorDict): Variables at which the subsampled Hessian
            is evaluated.
        X_batch (torch.Tensor): Subsampled data matrix at which the loss is evaluated.
        y_batch (torch.Tensor): Labels tensor at which the loss is evaluated.
        device (torch.device): Device the model parameters live on.

    Attributes:
        device (torch.device): Device on which computations are performed.
    """

    def __init__(
        self,
        loss: Callable[[TensorDict, tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
        variable_values: TensorDict,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        device: torch.device,
    ):
        """Initialize Subsampled Hessian linear operator.

        Args:
            loss (Callable): Loss function from which subsampled Hessian is constructed.
                Should have signature loss(variable_values, X, y) -> scalar tensor.
            variable_values (TensorDict): Variables at which the subsampled Hessian
                is evaluated.
            X_batch (torch.Tensor): Subsampled data matrix at which the loss is evaluated.
            y_batch (torch.Tensor): Labels tensor at which the loss is evaluated.
            device (torch.device): Device the model parameters live on.
        """
        super().__init__()
        self._loss = partial(loss, X=X_batch, y=y_batch)
        self._variable_values = variable_values

        n = variable_values.flat_dim()
        self._shape = (n, n)
        self.device = device

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        """Compute Hessian-vector product using forward-over-reverse autodiff.

        Computes the product of the subsampled Hessian matrix with a vector v
        without explicitly forming the Hessian matrix. Uses automatic differentiation
        to efficiently compute the Hessian-vector product.

        Args:
            v (torch.Tensor): Vector to multiply with the Hessian. Should have
                shape (n,) where n is the flattened dimension of variable_values.

        Returns:
            torch.Tensor: The result of Hessian @ v as a flattened tensor.
        """

        def grad_dot_v(var_vals: TensorDict) -> torch.Tensor:
            # Compute gradient of smooth_expr
            grad = torch.func.grad(lambda x: self._loss(x))(var_vals)
            return torch.dot(grad.to_flat_tensor(), v)

        # Differentiate grad_dot_v to get Hessian @ v
        hvp_td = torch.func.grad(grad_dot_v)(self._variable_values)
        return hvp_td.to_flat_tensor()
