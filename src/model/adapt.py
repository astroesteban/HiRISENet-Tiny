"""This module provides the functionality to modify user-provided PyTorch
models to work with the HiRISE v3.2 dataset.
"""

import torch
import torchvision


def get_modified_alexnet() -> torch.nn.Module:
    """Downloads and modifies the pre-trained `AlexNet` model for the HiRISE
    v3.2 dataset.

    The modifications consist of changing the input layer to support single
    channel greyscale images. Additionally, the function modifies the final
    output layer to predict from HiRISE's 8 classes instead of IMAGENET's
    1000 classes.

    Returns:
        torch.nn.Module: The modified model
    """
    model = torchvision.models.alexnet(
        weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    )

    # the number of channels in greyscale images
    num_channels: int = 1
    num_classes: int = 8

    # Extract the first conv layer's parameters
    num_filters: int = model.features[0].out_channels
    kernel_size: tuple = model.features[0].kernel_size
    stride: tuple = model.features[0].stride
    padding: tuple = model.features[0].padding

    # initialize a new convolutional layer
    conv1 = torch.nn.Conv2d(
        num_channels,
        num_filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    # Initialize the new conv1 layer's weights by averaging the pretrained weights across the channel dimension
    original_weights = model.features[0].weight.data.mean(dim=1, keepdim=True)
    # Expand the averaged weights to the number of input channels of the new dataset
    conv1.weight.data = original_weights.repeat(1, num_channels, 1, 1)

    model.features[0] = conv1

    # Modify the number of classes
    model.classifier[-1].out_features = num_classes

    try:
        test_tensor: torch.Tensor = torch.randn((1, 1, 227, 227))
        model(test_tensor)
    except RuntimeError:
        print("ERROR: Failed to adapt the model for the HiRISE dataset.")

    return model


def get_modified_resnet18() -> torch.nn.Module:
    """Downloads and modifies the pre-trained `ResNet18` model for the HiRISE
    v3.2 dataset.

    The modifications consist of changing the input layer to support single
    channel greyscale images. Additionally, the function modifies the final
    output layer to predict from HiRISE's 8 classes instead of IMAGENET's
    1000 classes.

    Returns:
        torch.nn.Module: The modified model
    """
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    )

    # the number of channels in greyscale images
    num_channels: int = 1
    num_classes: int = 8

    # Extract the first conv layer's parameters
    num_filters: int = model.conv1.out_channels
    kernel_size: tuple = model.conv1.kernel_size
    stride: tuple = model.conv1.stride
    padding: tuple = model.conv1.padding

    # initialize a new convolutional layer
    conv1 = torch.nn.Conv2d(
        num_channels,
        num_filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    # Initialize the new conv1 layer's weights by averaging the pretrained
    # weights across the channel dimension
    original_weights: torch.Tensor = model.conv1.weight.data.mean(dim=1, keepdim=True)

    # Expand the averaged weights to the number of input channels of the new dataset
    conv1.weight.data = original_weights.repeat(1, num_channels, 1, 1)

    model.conv1 = conv1

    # Modify the number of classes
    model.fc.out_features = num_classes

    try:
        test_tensor: torch.Tensor = torch.randn((1, 1, 227, 227))
        model(test_tensor)
    except RuntimeError:
        print("ERROR: Failed to adapt the model for the HiRISE dataset.")

    return model


def get_modified_efficientnet_b0() -> torch.nn.Module:
    """Downloads and modifies the pre-trained `EfficientNet B0` model for the
    HiRISE v3.2 dataset.

    The modifications consist of changing the input layer to support single
    channel greyscale images. Additionally, the function modifies the final
    output layer to predict from HiRISE's 8 classes instead of IMAGENET's
    1000 classes.

    Returns:
        torch.nn.Module: The modified model
    """
    model = torchvision.models.efficientnet_b0(
        weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    )

    # the number of channels in greyscale images
    num_channels: int = 1
    num_classes: int = 8

    # Extract the first conv layer's parameters
    num_filters: int = model.features[0][0].out_channels
    kernel_size: tuple = model.features[0][0].kernel_size
    stride: tuple = model.features[0][0].stride
    padding: tuple = model.features[0][0].padding

    # initialize a new convolutional layer
    conv1 = torch.nn.Conv2d(
        num_channels,
        num_filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    # Initialize the new conv1 layer's weights by averaging the pretrained weights across the channel dimension
    original_weights: torch.Tensor = model.features[0][0].weight.data.mean(
        dim=1, keepdim=True
    )

    # Expand the averaged weights to the number of input channels of the new dataset
    conv1.weight.data = original_weights.repeat(1, num_channels, 1, 1)

    model.features[0][0] = conv1

    # Modify the model from 1,000 classes to 8
    model.classifier[-1].out_features = num_classes

    try:
        test_tensor: torch.Tensor = torch.randn((1, 1, 227, 227))
        model(test_tensor)
    except RuntimeError:
        print("ERROR: Failed to adapt the model for the HiRISE dataset.")

    return model


def get_modified_efficientnet_v2_s() -> torch.nn.Module:
    """Downloads and modifies the pre-trained `EfficientNet v2 s` model for the
    HiRISE v3.2 dataset.

    The modifications consist of changing the input layer to support single
    channel greyscale images. Additionally, the function modifies the final
    output layer to predict from HiRISE's 8 classes instead of IMAGENET's
    1000 classes.

    Returns:
        torch.nn.Module: The modified model
    """
    model = torchvision.models.efficientnet_v2_s(
        weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    )

    # the number of channels in greyscale images
    num_channels: int = 1
    num_classes: int = 8

    # Extract the first conv layer's parameters
    num_filters: int = model.features[0][0].out_channels
    kernel_size: tuple = model.features[0][0].kernel_size
    stride: tuple = model.features[0][0].stride
    padding: tuple = model.features[0][0].padding

    # initialize a new convolutional layer
    conv1 = torch.nn.Conv2d(
        num_channels,
        num_filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    # Initialize the new conv1 layer's weights by averaging the pretrained weights across the channel dimension
    original_weights: torch.Tensor = model.features[0][0].weight.data.mean(
        dim=1, keepdim=True
    )

    # Expand the averaged weights to the number of input channels of the new dataset
    conv1.weight.data = original_weights.repeat(1, num_channels, 1, 1)

    model.features[0][0] = conv1

    # Modify the model from 1,000 classes to 8
    model.classifier[-1].out_features = num_classes

    try:
        test_tensor: torch.Tensor = torch.randn((1, 1, 227, 227))
        model(test_tensor)
    except RuntimeError:
        print("ERROR: Failed to adapt the model for the HiRISE dataset.")

    return model


def get_modified_convnext_tiny() -> torch.nn.Module:
    """Downloads and modifies the pre-trained `ConvNeXt Tiny` model for the
    HiRISE v3.2 dataset.

    The modifications consist of changing the input layer to support single
    channel greyscale images. Additionally, the function modifies the final
    output layer to predict from HiRISE's 8 classes instead of IMAGENET's
    1000 classes.

    Returns:
        torch.nn.Module: The modified model
    """
    model = torchvision.models.convnext_tiny(
        weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    )

    # the number of channels in greyscale images
    num_channels: int = 1
    num_classes: int = 8

    # Extract the first conv layer's parameters
    num_filters: int = model.features[0][0].out_channels
    kernel_size: tuple = model.features[0][0].kernel_size
    stride: tuple = model.features[0][0].stride
    padding: tuple = model.features[0][0].padding

    # initialize a new convolutional layer
    conv1 = torch.nn.Conv2d(
        num_channels,
        num_filters,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    # Initialize the new conv1 layer's weights by averaging the pretrained weights across the channel dimension
    original_weights: torch.Tensor = model.features[0][0].weight.data.mean(
        dim=1, keepdim=True
    )

    # Expand the averaged weights to the number of input channels of the new dataset
    conv1.weight.data = original_weights.repeat(1, num_channels, 1, 1)

    model.features[0][0] = conv1

    # Modify the model from 1,000 classes to 8
    model.classifier[-1].out_features = num_classes

    try:
        test_tensor: torch.Tensor = torch.randn((1, 1, 227, 227))
        model(test_tensor)
    except RuntimeError:
        print("ERROR: Failed to adapt the model for the HiRISE dataset.")

    return model
