"""This module provides useful utilities and tools to profile and understand
a PyTorch model.
"""

import torch
from matplotlib import pyplot as plt
from torch.utils.flop_counter import FlopCounterMode
from typing import Union, Tuple


def calc_model_size(model: torch.nn.Module) -> float:
    """Calculates the total size, in megabytes, of a
    model

    Args:
        model (torch.nn.Module): The PyTorch model
        show (bool, optional): Flag to print the size. Defaults to True.

    Returns:
        int: The size of the model in megabytes
    """
    param_size: int = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    return (param_size + buffer_size) / 1024**2


def show_model_sizes(models: dict[str, torch.nn.Module]) -> None:
    """Calculates and plots the model's sizes in megabytes

    Args:
        models (dict[str, torch.nn.Module]): A dictionary of models
    """
    model_sizes: list = list(map(calc_model_size, models.values()))

    plt.bar(models.keys(), model_sizes, width=-0.8, color="#EC407A")

    plt.title("Model Sizes")
    plt.xlabel("Models")
    plt.ylabel("Size (MB)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(model_sizes)
    plt.show()


def get_flops(
    model: torch.nn.Module, inp: Union[torch.Tensor, Tuple], with_backward=False
) -> int:
    """Calculates the number of floating point operations the model performs

    Args:
        model (torch.nn.Module): The model to calculate the FLOPS for
        inp (Union[torch.Tensor, Tuple]): The input to pass through the model
        with_backward (bool, optional): Do a backward pass. Defaults to False.

    Returns:
        int: The number of FLOPS the model performed on the input `inp`
    """
    is_train: bool = model.training
    model.eval()

    inp: torch.Tensor = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(display=False, depth=None)

    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)

    total_flops: int = flop_counter.get_total_flops()

    if is_train:
        model.train()

    return total_flops
