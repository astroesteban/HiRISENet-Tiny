from tempfile import TemporaryDirectory
import time
import torch
from pathlib import Path
from data.split_type import SplitType

# TODO: REMOVE
dataset_sizes = {SplitType.TRAIN: 51058, SplitType.VAL: 14959, SplitType.TEST: 1793}


def __train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, int]:
    """Trains a PyTorch model for one epoch

    Args:
        model (torch.nn.Module): The model to train
        dataloader (torch.utils.data.DataLoader): The dataloader for the test set
        criterion (torch.nn.Module): The loss function
        optimizer (torch.optim.Optimizer): The optimization function
        device (str): The device to compute on

    Returns:
        tuple[float, int]: The running loss and running number of correct predctions
    """
    model.train()  # Set model to training mode
    running_loss: float = 0.0
    running_corrects: int = 0

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs: torch.Tensor = inputs.to(device)
        labels: torch.Tensor = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)  # ? why do we need this?
        loss = criterion(outputs, labels)

        # backward + optimize
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    return running_loss, running_corrects


def __validate_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: str,
) -> tuple[float, int]:
    """Validates a PyTorch model for one epoch

    Args:
        model (torch.nn.Module): The model to train
        dataloader (torch.utils.data.DataLoader): The dataloader for the validation set
        criterion (torch.nn.Module): The loss function
        device (str): The device to compute on

    Returns:
        tuple[float, int]: The running loss and running number of correct predctions
    """
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # forward
        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    return running_loss, running_corrects


def train_model(
    model: torch.nn.Module,
    dataloaders: dict[str, torch.utils.data.DataLoader],
    criterion: torch.nn.Module,
    optimizer: torch.optim,
    scheduler: torch.optim.lr_scheduler,
    device: str,
    num_epochs: int = 3,
) -> tuple[torch.nn.Module, list[list[float]]]:
    """Trains a PyTorch model for the number of specified epochs

    Args:
        model (torch.nn.Module): The model to train
        dataloaders (dict[str, torch.utils.data.DataLoader]): The dataloaders containing our training data
        criterion (torch.nn.Module): The loss function
        optimizer (torch.optim): The optimization function
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler
        device (str): The device to do the training on
        num_epochs (int, optional): The number of epochs to train. Defaults to 3.

    Returns:
        tuple[torch.nn.Module, list[list[float]]]: The best model and training history
    """
    since: float = time.time()

    model.to(device)

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path: Path = Path(tempdir) / "best_model_params.pt"
        torch.save(model.state_dict(), best_model_params_path)
        best_acc: float = 0.0
        history: list = []

        for epoch in range(num_epochs):
            # Each epoch has a training and validation phase
            train_loss, train_corrects = __train_epoch(
                model, dataloaders[SplitType.TRAIN], criterion, optimizer, device
            )
            if scheduler:
                scheduler.step()
            val_loss, val_corrects = __validate_epoch(
                model, dataloaders[SplitType.VAL], criterion, device
            )

            train_loss /= dataset_sizes[SplitType.TRAIN]
            train_acc = train_corrects.double() / dataset_sizes[SplitType.TRAIN]
            val_loss /= dataset_sizes[SplitType.VAL]
            val_acc = val_corrects.double() / dataset_sizes[SplitType.VAL]

            history.append([train_acc, val_acc, train_loss, val_loss])
            print(
                f"Epoch {epoch}/{num_epochs - 1}: "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
            )

            # deep copy the model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_params_path)

        time_elapsed: float = time.time() - since
        print(
            f"\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s"
        )
        print(f"Best val Acc: {best_acc:.4f}")

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))

    return model, history
