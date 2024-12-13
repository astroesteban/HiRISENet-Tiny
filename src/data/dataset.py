"""PyTorch dataset class for loading the NASA HiRISE dataset"""

import torch
from typing import Any, Union, Optional, Callable, LiteralString
from pathlib import Path
from split_type import SplitType
from zenodo import download_and_extract
import pandas as pd
import csv
from PIL import Image


class HiRISE(torch.utils.data.Dataset):
    """NASA Mars HiRISE v3.2 Dataset
    https://zenodo.org/records/4002935
    """

    zenodo_hirise_record_id: str = "4002935"

    dataset_zip_file: tuple[Path, LiteralString] = (
        Path("hirise-map-proj-v3_2.zip"),
        "236d9c627db1a5970e77a01a8c8a035a",  # md5
    )

    image_folder: Path = Path(Path(dataset_zip_file[0].stem) / Path("map-proj-v3_2"))

    class_label_file: Path = Path(
        Path(dataset_zip_file[0].stem) / Path("labels-map-proj_v3_2.txt")
    )

    trained_model_labels_file: Path = Path(
        Path(dataset_zip_file[0].stem) / Path("labels-map-proj_v3_2_train_val_test.txt")
    )

    label_semantic_names_file: Path = Path(
        Path(dataset_zip_file[0].stem) / Path("landmarks_map-proj-v3_2_classmap.csv")
    )

    def __init__(
        self,
        root_dir: Union[str, Path],
        split_type: SplitType,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """NASA Mars HiRISE v3.2 Dataset
        https://zenodo.org/records/4002935

        Args:
            root_dir (Union[str, Path]): Root directory of dataset where
            `hirise-map-proj-v3_2.zip` exist.

            split_type (SplitType): Creates a dataset of the specified split
            type based on the `labels-map-proj-v3_2_train_val_test.txt` file.

            transform (Optional[Callable], optional):  A function/transform
            that takes in a PIL image and returns a transformed version.
            E.g, `transforms.RandomCrop`. Defaults to None.

            target_transform (Optional[Callable], optional):  A function
            that takes in a label and returns a transformed version.
            Defaults to None.

            function/transform that takes in the target and transforms it.
            Defaults to None.

            download (bool, optional): If True, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again. Defaults to False.
        """
        super().__init__()
        self.root_dir: Path = Path(root_dir)
        self.split_type: SplitType = split_type
        self.transform: Optional[Callable] = transform
        self.target_transform: Optional[Callable] = target_transform

        if download:
            self.download()

        if not self.__check_exists():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it"
            )

        self.targets: Any = self.__load_targets()
        self.class_map: dict[int, str] = self.__load_class_map()

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[Any, int]:
        """Loads a single image from the dataset

        Args:
            idx (int): The index of the image to load

        Returns:
            tuple[Any, int]: The image and target pair
        """
        img_file_name: Path = Path(
            self.raw_folder / self.image_folder / Path(self.targets.iloc[idx, 0])
        )

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_file_name)

        target: int = int(self.targets.iloc[idx, 1])

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    @property
    def raw_folder(self) -> Path:
        """The path to the raw data folder to download the mirror to

        Returns:
            Path: The path to the raw data folder
        """
        return Path(Path(self.root_dir) / Path(self.__class__.__name__) / "raw")

    def download(self) -> None:
        """Downloads the Mars orbital image (HiRISE) labeled data set version
        3.2 and unzips it
        """
        if self.__check_exists():
            return

        self.raw_folder.mkdir(parents=True, exist_ok=True)

        download_and_extract(
            record_id=self.zenodo_hirise_record_id,
            download_folder=self.raw_folder,
            md5=self.dataset_zip_file[1],
            filename=self.dataset_zip_file[0],
        )

    def __check_exists(self) -> bool:
        """Checks the filesystem for the presence of the HiRISE dataset

        Returns:
            bool: True if the unzipped dataset exists on disk.
        """
        # ! For now just check if the folder is unzipped. Add more checks later
        print(f"PATH: {Path(self.raw_folder / self.dataset_zip_file[0].stem)}")
        return Path(self.raw_folder / self.dataset_zip_file[0].stem).exists()

    def __load_targets(self) -> Any:
        """Loads the labels of the dataset.

        For the train, val, and test splits it will load the same splits the
        NASA team used to train their model i.e. the
        `labels-map-proj_v3_2_train_val_test.txt` file. Otherwise, if all is
        specified the function will load all the labels from
        `labels-map-proj_v3_2.txt`

        Returns:
            Any: _description_
        """
        if self.split_type != SplitType.ALL:
            df = pd.read_csv(
                self.raw_folder / self.trained_model_labels_file,
                sep=" ",
                header=None,
                names=["filename", "class_id", "set"],
                dtype={"filename": str, "class_id": int, "set": str},
                low_memory=False,
            )

            return df[df.set == self.split_type.name.lower()].drop("set", axis=1)
        else:
            return pd.read_csv(
                self.raw_folder / self.class_label_file,
                sep=" ",
                header=None,
                names=["filename", "class_id"],
                dtype={"filename": str, "class_id": int},
                low_memory=False,
            )

    def __load_class_map(self) -> dict[int, str]:
        """Loads the mapping from the class to the human-readable string

        Returns:
            dict[int, str]: The class map
        """
        with Path(self.raw_folder / self.label_semantic_names_file).open() as csv_file:
            reader = csv.reader(csv_file)
            return {int(rows[0]): rows[1] for rows in reader}
