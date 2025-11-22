import torch

from rlaopt.atoms.linear_model_base.base import _BaseLinearModel
from rlaopt.ext_tensordict import TensorDict


class BaseClassifier(_BaseLinearModel):
    """Linear model for classification tasks."""

    def __init__(self, loss_type, beta, dataloader, fit_intercept=True, **loss_kwargs):
        super().__init__(loss_type, beta, dataloader, fit_intercept, **loss_kwargs)

    def decision_function(
        self, beta_value: TensorDict | None = None, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute decision function scores.

        Args:
            beta_value: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.

        Returns:
            Decision function scores
        """
        beta_tensor, intercept_tensor = self._get_params(beta_value)
        return BaseClassifier._get_raw_prediction(
            beta_tensor,
            intercept_tensor,
            self.dataloader,
            X=X,
            fit_intercept=self.fit_intercept,
        )

    def predict(
        self, beta_value: TensorDict | None = None, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict class labels.

        Args:
            beta_value: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.

        Returns:
            Predicted class labels
        """
        probs = self.predict_proba(beta_value, X)
        return torch.argmax(probs, dim=1)

    def predict_proba(
        self, beta_value: TensorDict | None = None, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict class probabilities.

        Args:
            beta_value: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.

        Returns:
            Class probabilities
        """
        logits = self.decision_function(beta_value, X)
        if logits.dim() == 1:
            # Binary classification
            pos_probs = torch.sigmoid(logits)
            neg_probs = 1 - pos_probs
            return torch.stack([neg_probs, pos_probs], dim=1)
        else:
            # Multiclass classification
            return torch.softmax(logits, dim=1)

    def predict_log_proba(
        self, beta_value: TensorDict | None = None, X: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict log class probabilities.

        Args:
            beta_value: TensorDict storing value of model weights
            X: Optional input data. If None, uses training dataset.

        Returns:
            Log class probabilities
        """
        logits = self.decision_function(beta_value, X)
        if logits.dim() == 1:
            # Binary classification: return log probs for both classes
            log_pos_probs = torch.nn.functional.logsigmoid(logits)
            log_neg_probs = torch.nn.functional.logsigmoid(
                -logits
            )  # Using identity: log(1 - sigmoid(x)) = logsigmoid(-x)
            return torch.stack([log_neg_probs, log_pos_probs], dim=1)
        else:
            # Multiclass classification: use log_softmax
            return torch.nn.functional.log_softmax(logits, dim=1)

    def score(
        self,
        beta_value: TensorDict | None = None,
        X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
    ) -> float:
        """Compute classification accuracy."""
        y = self._get_target_values(X, y)
        y_pred = self.predict(beta_value, X)
        return (y_pred == y).float().mean().item()
