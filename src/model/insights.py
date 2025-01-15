"""This module provides useful utilities and tools to profile and understand
a PyTorch model.
"""

import torch
from torch.utils.flop_counter import FlopCounterMode
from typing import Union, Tuple
import time
import statistics


def flops(
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
    model = model.to("cpu")
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


def model_size(model: torch.nn.Module) -> None:
    """Calculates the number of parameters and buffers, multiplies them with the
    element size, and accumulated them. This is more reliable than depending
    on the size of the model on disk since that can vary with things like
    compression.
    Args:
        model (torch.nn.Module): The model to measure
    """
    param_size: int = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size: int = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb: float = (param_size + buffer_size) / 1024**2
    print("model size: {:.3f} MB".format(size_all_mb))


def inference_latency(model: torch.nn.Module, input_data: torch.Tensor) -> None:
    """Calculates the inference latency for our model.

    Inference latency can be noisy so this function calculates it in 3 stages:

    1. Do a simple forward pass

    2. Do a warm up run plus multiple passes

    3. Do a pass on the GPU

    Args:
        model (torch.nn.Module): The model to do the inference latency calcs
    """
    with torch.no_grad():
        model = model.to("cpu")
        model.eval()

        input_data = input_data.to("cpu")

        # Warm-up runs
        for _ in range(10):
            _ = model(input_data)

        # Multiple iterations
        num_iterations: int = 1000
        inference_times: list[float] = []
        for _ in range(num_iterations):
            start_time: float = time.time()
            _ = model(input_data)
            end_time: float = time.time()
            inference_times.append(end_time - start_time)
        average_inference_time = statistics.mean(inference_times)
        print(
            f"Average CPU inference time with warm-up: {average_inference_time:.4f} seconds"
        )

    if torch.cuda.is_available():
        with torch.no_grad():
            model = model.to("cuda")
            model.eval()

            input_data = input_data.to("cuda")
            # Warm-up runs
            for _ in range(10):
                _ = model(input_data)

            # Measure inference time with GPU synchronization
            num_iterations: int = 1000
            inference_times: list[float] = []

            for _ in range(num_iterations):
                torch.cuda.synchronize()
                start_time: float = time.time()
                _ = model(input_data)
                torch.cuda.synchronize()
                end_time: float = time.time()
                inference_times.append(end_time - start_time)

            average_inference_time = statistics.mean(inference_times)
            print(
                f"Average GPU Inference time with GPU synchronization and warm-up: {average_inference_time:.4f} seconds"
            )
