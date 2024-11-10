import importlib
from typing import List

import torch


def get_optimizer(
    optimizer_name: str,
    optimizer_params: dict,
    lr: float,
    model: torch.nn.Module = None,
    parameters: List = None,
) -> torch.optim.Optimizer:
    """Find, initialize and return a Torch optimizer.

    Args:
        optimizer_name (str): Optimizer name.
        optimizer_params (dict): Optimizer parameters.
        lr (float): Initial learning rate.
        model (torch.nn.Module): Model to pass to the optimizer.

    Returns:
        torch.optim.Optimizer: Functional optimizer.
    """
    if model is not None:
        parameters = model.parameters()

    if optimizer_name.lower() == "radam":
        module = importlib.import_module("optimizers.radam")
        optimizer = getattr(module, "RAdam")
    elif optimizer_name.lower() == "palm_soap":
        from heavyball import PaLMForeachSOAP

        module = PaLMForeachSOAP
    elif optimizer_name.lower() == "ademamix":
        module = importlib.import_module("optimizers.ademamix")
        optimizer = getattr(module, "AdEMAMix")
    elif optimizer_name.lower() == "adopt":
        module = importlib.import_module("optimizers.adopt")
        optimizer = getattr(module, "ADOPT")
        return optimizer(parameters, lr=lr, **optimizer_params, decoupled=True)
    else:
        optimizer = getattr(torch.optim, optimizer_name)
    return optimizer(parameters, lr=lr, **optimizer_params)
