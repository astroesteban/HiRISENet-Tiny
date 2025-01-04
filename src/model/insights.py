"""This module provides useful utilities and tools to profile and understand
a PyTorch model.
"""

import torch
import torchprofile
from matplotlib import pyplot as plt
from torch.utils.flop_counter import FlopCounterMode
from typing import Union, Tuple
from torch.profiler import profile, record_function, ProfilerActivity


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


def plot_model_sizes(models: dict[str, torch.nn.Module]) -> None:
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
    model: torch.nn.Module,
    inp: Union[torch.Tensor, Tuple] = (1, 1, 227, 227),
    with_backward=False,
) -> int:
    """Calculates the number of floating point operations the model performs

    Args:
        model (torch.nn.Module): The model to calculate the FLOPS for
        inp (Union[torch.Tensor, Tuple]): The input to pass through the model.
        Defaults to (1, 1, 227, 227).
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


def plot_model_flops(models: dict[str, torch.nn.Module]) -> None:
    """Calculates and plots the model's FLOPS

    Args:
        models (dict[str, torch.nn.Module]): A dictionary of models
    """
    model_flops: list = list(map(get_flops, models.values()))

    plt.bar(models.keys(), model_flops, width=-0.8, color="#EC407A")

    plt.title("Model FLOPS")
    plt.xlabel("Models")
    plt.ylabel("FLOPS")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(model_flops)
    plt.show()


def plot_model_macs(models: dict[str, torch.nn.Module]) -> None:
    """Calculates and plots the model's MACs

    Args:
        models (dict[str, torch.nn.Module]): A dictionary of models
    """
    input_tensor: torch.Tensor = torch.randn(1, 1, 227, 227)

    macs: list = [
        torchprofile.profile_macs(model, input_tensor) for model in models.values()
    ]

    plt.bar(models.keys(), macs, width=-0.8, color="#EC407A")

    plt.title("Model MACs")
    plt.xlabel("Models")
    plt.ylabel("MACs")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(macs)
    plt.show()


def plot_cpu_execution_time(models: dict[str, torch.nn.Module]) -> None:
    """Calculates and plots the model's CPU execution time on inference

    Args:
        models (dict[str, torch.nn.Module]): A dictionary of models
    """
    inputs: torch.Tensor = torch.randn((1, 1, 227, 227))

    execution_times: list[float] = []

    for model in models.values():
        model.eval()
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with record_function("model_inference"):
                model(inputs)

        # convert the execution time to milliseconds
        execution_times.append(prof.key_averages()[0].cpu_time / 1000)

    plt.bar(models.keys(), execution_times, width=-0.8, color="#EC407A")

    plt.title("Model Execution Times (CPU)")
    plt.xlabel("Models")
    plt.ylabel("Execution Time (ms)")
    plt.xticks(rotation=45, ha="right")
    # plt.yticks(execution_times)
    plt.show()


def plot_gpu_execution_time(models: dict[str, torch.nn.Module]) -> None:
    """Calculates and plots the model's GPU execution time on inference

    Args:
        models (dict[str, torch.nn.Module]): A dictionary of models
    """
    # ! FIXME: This is not showing GPU metrics in WSL
    inputs: torch.Tensor = torch.randn((1, 1, 227, 227), device="cuda")

    execution_times: list[float] = []

    for model in models.values():
        model.eval()
        model.to("cuda")
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            with record_function("model_inference"):
                model(inputs)

        # convert the execution time to milliseconds
        execution_times.append(prof.key_averages()[0].cpu_time / 1000)

    plt.bar(models.keys(), execution_times, width=-0.8, color="#EC407A")

    plt.title("Model Execution Times (GPU)")
    plt.xlabel("Models")
    plt.ylabel("Execution Time (ms)")
    plt.xticks(rotation=45, ha="right")
    # plt.yticks(execution_times)
    plt.show()


def plot_max_mem_usage(models: dict[str, torch.nn.Module]) -> None:
    """Calculates the memory utilization of each layer of the model and plots
    the peak memory utilization.

    Args:
        models (dict[str, torch.nn.Module]): A dictionary of models
    """
    inputs = torch.randn(1, 1, 227, 227)

    max_ram: list[float] = []

    for model in models.values():
        model.eval()
        model.to("cpu")
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            model(inputs)

        # convert from bytes to megabytes
        max_layer_mem_usage: float = (
            max(node.cpu_memory_usage for node in prof.key_averages()) / 1000000
        )

        max_ram.append(max_layer_mem_usage)

    plt.bar(models.keys(), max_ram, width=-0.8, color="#EC407A")

    plt.title("Peak Layer Memory Usage (CPU)")
    plt.xlabel("Models")
    plt.ylabel("Memory Usage (Mb)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(max_ram)
    plt.show()
