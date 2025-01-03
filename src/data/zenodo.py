"""This module provides utilities for interacting with Zenodo"""

import requests
from typing import Any, Union, LiteralString
from pathlib import Path
from hashlib import md5

from tqdm import tqdm
import zipfile

from mmap import mmap, ACCESS_READ


def __extract_download(file: Path) -> None:
    """Extracts the input file if applicable i.e. it is a zip or tarball

    Args:
        file (Path): The file to extract
    """
    file = Path(file)

    if file.suffix == ".zip":
        with zipfile.ZipFile(file) as zip_ref:
            zip_ref.extractall(file.parent)


def __calculate_md5_checksum(file: Path) -> str:
    """Iteratively calculates the MD5 checksum of the input file

    Args:
        file (Path): The full path to the file to calculate the checksum on

    Returns:
        str: The MD5 checksum hex digest
    """
    with open(file, "rb") as file:
        with mmap(file.fileno(), 0, access=ACCESS_READ) as mmapped_file:
            return md5(mmapped_file, usedforsecurity=False).hexdigest()


def download_and_extract(
    record_id: str,
    download_folder: Union[str, Path],
    md5: LiteralString,
    filename: Union[str, Path],
    access_token: str = "",
) -> None:
    """This function downloads an artifact from the Zenodo repository. It will
    match the filename from the Zenodo record repository and download just the
    one file.

    Args:
        record_id (str): The unique record identifier to query. You can usually
        find this in the URL e.g. `https://zenodo.org/records/4002935`.

        download_folder (Union[str, Path]): The path to download the artifacts
        to.

        md5 (LiteralString): The MD5 hash of the file to download.

        filenames (Union[str, Path]): The name of the file to download.

        access_token (str, optional): The Zenodo access token typically used to
        download artifacts from private Zenodo repositories. Defaults to "".
    """
    full_file_path: Path = Path(Path(download_folder) / Path(filename))
    zenodo_url: str = (
        f"https://zenodo.org/api/records/{record_id}/files/{filename}/content"
    )

    # TODO: Refactor this to use multiple jobs to download the dataset

    # First lets download the file
    with full_file_path.open(mode="wb") as file:
        try:
            r = requests.get(
                zenodo_url, params={"access_token": access_token}, stream=True
            )
            r.raise_for_status()
        except requests.exceptions.RequestException as errex:
            print(errex)
            return

        num_bytes: int = int(r.headers.get("content-length", 0))

        # tqdm has many interesting parameters. Feel free to experiment!
        tqdm_params: dict[str, Any] = {
            "desc": zenodo_url,
            "total": num_bytes,
            "miniters": 1,
            "unit": "B",
            "unit_scale": True,
            "unit_divisor": 1024,
        }

        with tqdm(**tqdm_params) as pb:
            # ? Can we calculate the md5 hash as we download? How?
            for chunk in r.iter_content(chunk_size=8192):
                pb.update(len(chunk))
                file.write(chunk)

    if not full_file_path.exists():
        raise RuntimeError(
            f"The expected downloaded file {full_file_path} does not exist."
        )

    # Now lets verify that we downloaded it correctly
    if __calculate_md5_checksum(full_file_path) != md5:
        raise RuntimeError(
            "ERROR: The expected checksum doesn't match the calculated checksum. The downloaded file may be corrupted."
        )

    # Finally lets extract the file
    __extract_download(full_file_path)
