"""This module provides the functionality to modify user-provided PyTorch
models to work with the HiRISE v3.2 dataset.
"""

import torch
import os
import requests
from typing import Any
from tqdm import tqdm
from pathlib import Path
from .efficientnet_lite import build_efficientnet_lite
from enum import StrEnum


class EfficientNetVariant(StrEnum):
    """An enumeration containing the EfficientNet Lite variants"""

    Lite0 = "efficientnet_lite0"
    Lite1 = "efficientnet_lite1"
    Lite2 = "efficientnet_lite2"
    Lite3 = "efficientnet_lite3"
    Lite4 = "efficientnet_lite4"


def __download_weights(variant: EfficientNetVariant) -> Path:
    """Downloads the EfficientNet Lite IMAGENET model weights

    Args:
        name (str): The name of the model to download weights for.

    Returns:
        Path: The full path to the model weights.
    """
    weights_url: str = f"https://github.com/RangiLyu/EfficientNet-Lite/releases/download/v1.0/{variant}.pth"

    model_tarball: Path = Path(weights_url.split("/")[-1])

    download_folder: Path = Path(os.environ.get("TORCH_HOME", "/tmp/.torch"))
    download_folder.mkdir(parents=True, exist_ok=True)

    full_file_path: Path = download_folder / model_tarball

    # TODO: I should probably refactor this and the zenodo code to share code
    if not full_file_path.exists():
        with full_file_path.open(mode="wb") as file:
            try:
                r = requests.get(weights_url, stream=True)
                r.raise_for_status()
            except requests.exceptions.RequestException as errex:
                print(errex)
                return

            num_bytes: int = int(r.headers.get("content-length", 0))

            # tqdm has many interesting parameters. Feel free to experiment!
            tqdm_params: dict[str, Any] = {
                "desc": weights_url,
                "total": num_bytes,
                "miniters": 1,
                "unit": "B",
                "unit_scale": True,
                "unit_divisor": 1024,
            }

            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    file.write(chunk)

    if not full_file_path.exists():
        raise RuntimeError(
            f"The expected downloaded file {full_file_path} does not exist."
        )

    return full_file_path


def download_efficientnet_lite(
    variant: EfficientNetVariant = EfficientNetVariant.Lite0,
    num_channels: int = 1,
    num_classes: int = 8,
    pretrain: bool = True,
) -> torch.nn.Module:
    """Downloads and modifies the pre-trained `EfficientNet Lite` model for the
    HiRISE v3.2 dataset.

    The modifications consist of changing the input layer to support single
    channel greyscale images. Additionally, the function modifies the final
    output layer to predict from HiRISE's 8 classes instead of IMAGENET's
    1000 classes.

    Args:
        variant (EfficientNetVariant): The EfficientNet Lite variant to prepare. Default `EfficientNetVariant.Lite0`.
        num_channels (int): The number of channels in the input data. Default `1`.
        num_classes (int): The number of classes we want to predict. Default `8`.
        pretrain (bool): Download the pretrained model weights. Default `True`.

    Returns:
        torch.nn.Module: The EfficientNet Lite model
    """
    # First we need to create the model body in memory
    model: torch.nn.Module = build_efficientnet_lite(variant, 1000)

    # Load the pre-trained model weights
    if pretrain:
        weights_path: Path = __download_weights(variant)

        model.load_pretrain(weights_path)

    model.eval()

    model.adapt_to_data(num_channels, num_classes)

    # Modify the model from 1,000 classes to 8
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    try:
        test_tensor: torch.Tensor = torch.randn((1, 1, 227, 227))
        preds: torch.Tensor = model(test_tensor)
        assert preds.shape == (1, 8)
    except RuntimeError:
        print("ERROR: Failed to adapt the model for the HiRISE dataset.")

    return model


if __name__ == "__main__":
    download_efficientnet_lite()
