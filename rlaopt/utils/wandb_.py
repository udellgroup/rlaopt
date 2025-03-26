"""This module provides utility functions for working with Weights and Biases."""

import os


__all__ = ["set_wandb_api_key"]


def set_wandb_api_key(api_key: str):
    """Set the API key for Weights and Biases.

    Args:
        api_key (str): The API key provided by Weights and Biases.
    """
    os.environ["WANDB_API_KEY"] = api_key
