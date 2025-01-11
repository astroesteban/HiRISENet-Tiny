"""EfficientNet Lite Implementation
Original Source: https://github.com/RangiLyu/EfficientNet-Lite

I modified the original source code to facilitate doing things like modifying
the input channels and number of classes.
"""

import math
import torch
from torch import nn
import torch.functional as F
from pathlib import Path


__efficientnet_lite_params: dict[str, list[float]] = {
    # width_coefficient, depth_coefficient, image_size, dropout_rate
    "efficientnet_lite0": [1.0, 1.0, 224, 0.2],
    "efficientnet_lite1": [1.0, 1.1, 240, 0.2],
    "efficientnet_lite2": [1.1, 1.2, 260, 0.3],
    "efficientnet_lite3": [1.2, 1.4, 280, 0.3],
    "efficientnet_lite4": [1.4, 1.8, 300, 0.3],
}


class MBConvBlock(nn.Module):
    def __init__(self, inp, final_oup, k, s, expand_ratio, se_ratio, has_se=False):
        super(MBConvBlock, self).__init__()

        self._momentum = 0.01
        self._epsilon = 1e-3
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.has_se = has_se
        self.id_skip = True  # skip connection and drop connect

        # Expansion phase
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._momentum, eps=self._epsilon
            )

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,  # groups makes it depthwise
            kernel_size=k,
            padding=(k - 1) // 2,
            stride=s,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._momentum, eps=self._epsilon
        )

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(inp * se_ratio))
            self._se_reduce = nn.Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
            )
            self._se_expand = nn.Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
            )

        # Output phase
        self._project_conv = nn.Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._momentum, eps=self._epsilon
        )
        self._relu = nn.ReLU6(inplace=True)

    def __drop_connect(self, x, drop_connect_rate, training):
        if not training:
            return x
        keep_prob = 1.0 - drop_connect_rate
        batch_size = x.shape[0]
        random_tensor = keep_prob
        random_tensor += torch.rand(
            [batch_size, 1, 1, 1], dtype=x.dtype, device=x.device
        )
        binary_mask = torch.floor(random_tensor)
        x = (x / keep_prob) * binary_mask
        return x

    def forward(self, x, drop_connect_rate=None):
        """
        :param x: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            x = self._relu(self._bn0(self._expand_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._relu(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        if (
            self.id_skip
            and self.stride == 1
            and self.input_filters == self.output_filters
        ):
            if drop_connect_rate:
                x = self.__drop_connect(x, drop_connect_rate, training=self.training)
            x += identity  # skip connection
        return x


class EfficientNetLite(nn.Module):
    def __init__(
        self,
        widthi_multiplier,
        depth_multiplier,
        num_classes,
        drop_connect_rate,
        dropout_rate,
    ):
        super(EfficientNetLite, self).__init__()

        # Batch norm parameters
        momentum = 0.01
        epsilon = 1e-3
        self.drop_connect_rate = drop_connect_rate

        mb_block_settings = [
            # repeat|kernal_size|stride|expand|input|output|se_ratio
            [1, 3, 1, 1, 32, 16, 0.25],
            [2, 3, 2, 6, 16, 24, 0.25],
            [2, 5, 2, 6, 24, 40, 0.25],
            [3, 3, 2, 6, 40, 80, 0.25],
            [3, 5, 1, 6, 80, 112, 0.25],
            [4, 5, 2, 6, 112, 192, 0.25],
            [1, 3, 1, 6, 192, 320, 0.25],
        ]

        # Stem
        out_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(3, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )

        # Build blocks
        self.blocks = nn.ModuleList([])
        for i, stage_setting in enumerate(mb_block_settings):
            stage = nn.ModuleList([])
            (
                num_repeat,
                kernal_size,
                stride,
                expand_ratio,
                input_filters,
                output_filters,
                se_ratio,
            ) = stage_setting
            # Update block input and output filters based on width multiplier.
            input_filters = (
                input_filters
                if i == 0
                else self.__round_filters(input_filters, widthi_multiplier)
            )
            output_filters = self.__round_filters(output_filters, widthi_multiplier)
            num_repeat = (
                num_repeat
                if i == 0 or i == len(mb_block_settings) - 1
                else self.__round_repeats(num_repeat, depth_multiplier)
            )

            # The first block needs to take care of stride and filter size increase.
            stage.append(
                MBConvBlock(
                    input_filters,
                    output_filters,
                    kernal_size,
                    stride,
                    expand_ratio,
                    se_ratio,
                    has_se=False,
                )
            )
            if num_repeat > 1:
                input_filters = output_filters
                stride = 1
            for _ in range(num_repeat - 1):
                stage.append(
                    MBConvBlock(
                        input_filters,
                        output_filters,
                        kernal_size,
                        stride,
                        expand_ratio,
                        se_ratio,
                        has_se=False,
                    )
                )

            self.blocks.append(stage)

        # Head
        in_channels = self.__round_filters(mb_block_settings[-1][5], widthi_multiplier)
        out_channels = 1280
        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.fc = torch.nn.Linear(out_channels, num_classes)

        self._initialize_weights()

    def __round_filters(self, filters, multiplier, divisor=8, min_width=None):
        """Calculate and round number of filters based on width multiplier."""
        if not multiplier:
            return filters
        filters *= multiplier
        min_width = min_width or divisor
        new_filters = max(min_width, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def __round_repeats(self, repeats, multiplier):
        """Round number of filters based on depth multiplier."""
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def forward(self, x):
        x = self.stem(x)
        idx = 0
        for stage in self.blocks:
            for block in stage:
                drop_connect_rate = self.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.blocks)
                x = block(x, drop_connect_rate)
                idx += 1
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0 / float(n))
                m.bias.data.zero_()

    def load_pretrain(self, path: Path) -> None:
        """Loads the pre-trained ImageNet model weights

        Args:
            path (Path): The path to the weights state dictionary.
        """
        self.load_state_dict(torch.load(path, weights_only=True), strict=True)

    def adapt_to_data(self, num_channels: int, num_classes: int) -> None:
        """Modifies the model's input layers to match the number of expected
        channels. This is useful for going from RGB to B&W.
        Also modifies the model's output layer to match the number
        of classes.

        Args:
            num_channels (int): The number of input channels the images have.
            num_classes (int): The number of classes in our dataset.
        """
        # Modify the layer's number of expected image channels
        if num_channels != self.stem[0].in_channels:
            # Extract the first conv layer's parameters
            num_filters: int = self.stem[0].out_channels
            kernel_size: tuple = self.stem[0].kernel_size
            stride: tuple = self.stem[0].stride
            padding: tuple = self.stem[0].padding

            # initialize a new convolutional layer
            conv1 = torch.nn.Conv2d(
                num_channels,
                num_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

            # Initialize the new conv1 layer's weights by averaging the pretrained weights across the channel dimension
            original_weights: torch.Tensor = self.stem[0].weight.data.mean(
                dim=1, keepdim=True
            )

            # Expand the averaged weights to the number of input channels of the new dataset
            conv1.weight.data = original_weights.repeat(1, num_channels, 1, 1)

            self.stem[0] = conv1

        # Modify the model's number of classes in the fully-connected output layer
        if num_classes != self.fc.in_features:
            self.fc = torch.nn.Linear(self.fc.in_features, num_classes)


def build_efficientnet_lite(name, num_classes):
    width_coefficient, depth_coefficient, _, dropout_rate = __efficientnet_lite_params[
        name
    ]
    model = EfficientNetLite(
        width_coefficient, depth_coefficient, num_classes, 0.2, dropout_rate
    )
    return model
