from typing import Any


def calc_model_size(model: Any, show: bool = True) -> float:
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
        print("model size: {:.3f}MB".format(size_all_mb))

    return size_all_mb
