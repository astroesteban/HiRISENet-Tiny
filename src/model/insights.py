"""This module provides useful utilities and tools to profile and understand
a PyTorch model.
"""

from typing import Any
import torch
from matplotlib import pyplot as plt


def calc_model_size(model: Any, show: bool = False) -> float:
    """Calculates the total size, in megabytes, of a
    model

    Args:
        model (Any): The PyTorch model
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

    size_all_mb = (param_size + buffer_size) / 1024**2

    if show:
        print("Model Size: {:.3f} MB".format(size_all_mb))

    return size_all_mb


def show_model_sizes(models: dict[str, torch.nn.Module]) -> None:
    """_summary_

    Args:
        models (dict[str, torch.nn.Module]): _description_
    """
    model_sizes: list = list(map(calc_model_size, models.values()))

    plt.bar(models.keys(), model_sizes, width=-0.8, color="#EC407A")

    plt.title("Model Sizes")
    plt.xlabel("Models")
    plt.ylabel("Size (MB)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(model_sizes)
    plt.show()
