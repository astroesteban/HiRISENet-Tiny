import pytest
from dataset import HiRISE
from pathlib import Path
from split_type import SplitType
import torchvision.transforms as transforms


@pytest.fixture()
def data_path():
    return Path(Path(__file__).parent / Path("assets/data"))


@pytest.fixture()
def image_transformations():
    return transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()])


def test_dataset_correct_number_of_samples_loaded(data_path):
    dataset = HiRISE(root_dir=data_path, split_type=SplitType.TRAIN)
    assert len(dataset) == 1


def test_dataset_data_correct_image_shape(data_path):
    dataset = HiRISE(root_dir=data_path, split_type=SplitType.TRAIN)
    assert dataset[0][0].size == (227, 227)


def test_dataset_correct_target_loaded(data_path):
    dataset = HiRISE(root_dir=data_path, split_type=SplitType.TRAIN)
    assert dataset[0][1] == 7


def test_dataset_correct_class_map_loaded(data_path):
    dataset = HiRISE(root_dir=data_path, split_type=SplitType.TRAIN)
    assert dataset.class_map == {7: "spider"}


def test_dataset_apply_image_transforms(data_path, image_transformations):
    dataset = HiRISE(
        root_dir=data_path, split_type=SplitType.TRAIN, transform=image_transformations
    )
    assert dataset[0][0].shape == (1, 227, 227)
