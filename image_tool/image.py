import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile


@dataclass
class ImageKey:
    row: int
    col: int
    fov: int
    channel: int

    def __str__(self):
        return f"r{self.row}c{self.col}f{self.fov}ch{self.channel}.tiff"


class Image:
    def __init__(self):
        self.key: ImageKey = ImageKey(0, 0, 0, 0)
        self.time_point: int = 0
        self.image_path: Path = Path()

    def constr_raw_image(self, image_name: str, folder_path: str):
        self._parse_name(image_name)
        self._set_image_path(image_name, folder_path)

    def constr_parsed_image(self, key: ImageKey, image_path: Path) -> None:
        self.key = key
        self.image_path = image_path

    def _set_image_path(self, image_name: str, folder_path: str) -> None:
        self.image_path = Path(os.path.join(folder_path, image_name))

    def read_image(self) -> np.ndarray:
        return np.array(tifffile.imread(self.image_path.as_posix()))

    def _parse_name(self, image_name: str):
        """
        Parse the image name to get the row, col, fov, channel and time point
        """
        regex = r'r([0-9][0-9])c([0-9][0-9])f([0-9][0-9])p([0-9][0-9])-ch([0-9][0-9])t([0-9][0-9]).tiff'
        match = re.search(regex, image_name)
        if match is None:
            raise ValueError(f"Image name {image_name} does not match the required name format")

        groups = match.groups()

        if len(groups) != 5:
            raise ValueError(f"Image name {image_name} does not match the required name format")

        self.key = ImageKey(int(groups[0]), int(groups[1]), int(groups[2]), int(groups[3]))
        self.time_point = int(groups[4])
