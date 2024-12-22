"""This is a custom model for HiRISE images"""

import torch


class HiRISENet(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(HiRISENet, self).__init__()

        self.layer1 = torch.nn.Sequential(
            # input: bsx1x32x32
            torch.nn.Conv2d(
                in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0
            ),
            # input: bsx6x28x28
            torch.nn.BatchNorm2d(
                num_features=6
            ),  # ? In the original batchnorm paper they recommend applying batchnorm prior to relu. But recent evidence suggests its best after?
            # input: bsx6x28x28
            torch.nn.ReLU(),
            # input: bsx6x28x28
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
            ),
            # input: bsx16x10x10
            torch.nn.BatchNorm2d(num_features=16),
            # input: bsx16x10x10
            torch.nn.ReLU(),
            # input: bsx16x10x10
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # input: bsx16x5x5

        self.fc = torch.nn.Linear(
            400, 120
        )  # ? how did they get these numbers for the features? *(5x5x16)=400
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(120, 84)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.layer1(x)
        out = self.layer2(out)

        # flatten the tensor
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)

        return out
