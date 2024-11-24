"""This modules provides utilities for interacting with and managing the GPU"""

import torch
from typing import Any


class GPUManager:
    """This class provides utilities to manage the GPU"""

    @staticmethod
    def cleanup_gpu_cache() -> None:
        """Utility function to cleanup unused CUDA memory"""
        with torch.no_grad():
            torch.cuda.empty_cache()

        import gc

        gc.collect()

    @staticmethod
    def enable_gpu_if_available() -> Any:
        """Checks if a GPU is available and if it
        is then it prints out the GPU information
        and returns the device handle

        Returns:
            Any: The torch.device handle
        """
        # Enable GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            print("__CUDA VERSION:", torch.backends.cudnn.version())
            print("__Number CUDA Devices:", torch.cuda.device_count())
            print("__CUDA Device Name:", torch.cuda.get_device_name(0))
            print(
                "__CUDA Device Total Memory [GB]:",
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )

        return device
