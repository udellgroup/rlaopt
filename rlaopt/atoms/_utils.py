import torch


def get_variable_value(
    var: torch.nn.Parameter, **variable_locations
) -> torch.nn.Parameter:
    # Use substituted value if available, otherwise use registered variable
    if variable_locations and var.name in variable_locations:
        value = var.evaluate_at(**variable_locations)
    else:
        value = var

    return value
