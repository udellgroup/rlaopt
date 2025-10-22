"""Identity preconditioner and configuration."""

from rlaopt.linalg.preconditioners.preconditioner import (
    Preconditioner,
    PreconditionerConfig,
)


class IdentityConfig(PreconditionerConfig):
    """Configuration for the Identity preconditioner."""

    pass


class Identity(Preconditioner):
    """Identity preconditioner implementation."""

    def __init__(self, config: IdentityConfig):
        """Initialize the Identity preconditioner with the given configuration.

        Args:
            config (IdentityConfig): Configuration for the Identity preconditioner.
        """
        super().__init__(config)
