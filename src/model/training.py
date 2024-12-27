"""This module contains methods to train our PyTorch model"""

import torch


def fine_tune(
    model: torch.nn,
    loss: torch.nn,
    optimizer: torch.optim,
    epochs: int,
    lr_scheduler: torch.lr_scheduler,
    head: bool = True,
):
    pass
