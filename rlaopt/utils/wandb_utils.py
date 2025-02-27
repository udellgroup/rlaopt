import wandb


def set_wandb_api_key(wandb_api_key: str):
    """
    Set the wandb API key for logging. This function handles logging in
    wandb with the specified API key.

    Args:
        wandb_api_key (str): The wandb API key to log in with.
    """
    if wandb_api_key:
        wandb.login(key=wandb_api_key)
