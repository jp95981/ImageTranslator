import os
from collections import defaultdict
from pathlib import Path
from typing import Generator, List, Tuple, Dict, DefaultDict, Any

import cv2 as cv  # type: ignore
import numpy as np
import numpy.typing as npt
import scipy.signal as signal
import tifffile as tiff

from helper import NoneSelected, is_hoesht, group_all_images_in_round
from image import Image


class Converter:
    def __init__(self, data_loc: str = "./data") -> None:
        self.data_loc: Path = Path(data_loc)
        self.round_directories: List[
            str
        ] = []  # A list of all the paths for each round
        self.round_to_tiffs_dict: DefaultDict[str, List[Image]] = defaultdict(
            list
        )  # Mapping of the round to the image stacks
        self.raw_images: DefaultDict[str, List[Image]] = defaultdict(list)
        self.shifters: List[np.ndarray] = list()
        self.translations: Dict[str, Tuple[Any, Any]]

        self.parsed_images: List[Image]
        self.stacked_data_location: Path
        self._image_setup()

    def _image_setup(self):
        self._find_round_paths()
        self._parse_all_images()
        self._set_new_data_location()
        self._stack_images()
        self._find_translations()

    def _find_round_paths(self):
        """
        finds the paths for each round and stores them in self.round_directories
        :return:
        """
        self.round_directories = [
            x for x in os.listdir(self.data_loc) if not x.startswith(".")
        ]  # Remove any temp folders

        self.translations = {self.round_directories[0]: (0, 0)}

    def _parse_all_images(self):
        for round_ in self.round_directories:
            round_path = os.path.join(self.data_loc, round_)
            all_images_in_round = os.listdir(round_path)
            for image_name in all_images_in_round:
                image = Image()
                image.constr_raw_image(image_name, round_path)
                self.raw_images[round_].append(image)

    def _set_new_data_location(self):
        data_parent_path = Path(self.data_loc).parent.absolute()
        stacked_images_folder_name = 'stacked_images'
        data_new_path = Path.joinpath(data_parent_path, stacked_images_folder_name)
        data_new_path.mkdir(exist_ok=True)
        self.stacked_data_location = data_new_path

    @staticmethod
    def _get_image_cluster(
            images: List[Image], chunk_size: int = 200
    ) -> Generator[npt.NDArray, None, None]:
        for i in range(0, len(images), chunk_size):
            chunked_names = images[i: i + chunk_size]
            yield np.array(
                [
                    img.read_image()
                    for img in chunked_names
                ]
            )

    def _stack_images(self) -> None:
        """
        Stacks images of different FOV's together
        For each round, find all the images of the same well and stack them together
        """
        for round_ in self.raw_images:
            groups = group_all_images_in_round(self.round_to_tiffs_dict[round_])
            for key in groups:
                group = groups[key]
                tiff_name = key.__str__()
                max_image: npt.NDArray[np.float64] = np.zeros(1)
                for images in self._get_image_cluster(group):
                    # Any empty images
                    if images.shape[0] == 0:
                        continue

                    # Stack the images
                    max_image = np.max(images, axis=0)

                save_directory = self.stacked_data_location.joinpath(round_)
                file_path = save_directory.joinpath(tiff_name)
                tiff.imwrite(save_directory, max_image)

                stacked_image = Image()
                stacked_image.constr_parsed_image(key, file_path)
                self.round_to_tiffs_dict[round_].append(stacked_image)

                if is_hoesht(key):
                    self.shifters.append(max_image)

    @staticmethod
    def _cross_image(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
        """
        Calculate the convolution of two images to find the translation between them

        Args:
            im1 (np.ndarray): Base image, image we want im2 to be aligned with
            im2 (np.ndarray): Image to shift

        Returns:
            np.ndarray: Convolved image
        """
        im1 = im1 - np.mean(im1)
        im2 = im2 - np.mean(im2)
        return signal.fftconvolve(im1, im2[::-1, ::-1], mode="same")

    def _find_translations(self):
        """
        Find how much to translate each round by
        So translate rounds 2 onwards so that they line up with round 1 image
        """
        for i in range(1, len(self.shifters)):
            # Find correlation between the current image and the one from rnd 1
            corr = self._cross_image(self.shifters[0], self.shifters[i])
            # the brightest point of the convoled image is the new image center
            x, y = np.unravel_index(np.argmax(corr), corr.shape)
            # So the difference between the current image center and the brightest point
            # is the translation
            translation: Tuple[Any, Any] = (x - corr.shape[0] // 2, y - corr.shape[1] // 2)
            current_round = self.round_directories[i]
            self.translations[current_round] = translation
        del self.shifters  # clearing them out of memory to save space

    def shift_image(self, im1: np.ndarray, rnd: str) -> np.ndarray:
        """
        Given an image and a round, find how to shift the image and shift it

        Args:
            im1 (np.ndarray): Image to be shifted
            rnd (str): Round the image belongs to

        Returns:
            np.ndarray: Shifted image
        """
        x, y = self.translations[rnd]
        transform = np.float32([[1, 0, -x], [0, 1, -y]])  # type: ignore
        shifted = cv.warpAffine(im1, transform, (im1.shape[0], im1.shape[1]))
        return shifted

    def overlay(self, round_to_image_dict: Dict[str, List[Image]]) -> np.ndarray:
        """
        Find the selected images and overlay them (i.e. the maximum pixels)

        Args:
            round_to_image_dict (Dict[str, List[ImageName]]):
                Dictionary of all the images RoundName -> [ImagesNames]

        Raises:
            NoneSelected: If no images are selected

        Returns:
            np.ndarray: The overlaid image of all those selected
        """
        stacked = []
        for round_ in round_to_image_dict:
            for image in round_to_image_dict[round_]:
                image = image.read_image()
                if (
                        round_ != self.round_directories[0]
                ):  # If not in the first round, then w need to shift the image
                    image = self.shift_image(image, round_)
                stacked.append(image)

        if len(stacked) == 0:
            raise NoneSelected("No images selected")

        stack = np.stack(stacked, axis=0)
        max_stack = np.max(stack, axis=0)
        return max_stack

    @staticmethod
    def save(image: np.ndarray, name: str) -> None:
        """
        Save the image which is passed in under the given name, in a folder called 'saves'

        Args:
            image (np.ndarray): Image to save
            name (str): Name to save it under

        Raises:
            NoneSelected: If the image is of 'None' type
        """
        if image is None:
            raise NoneSelected("No images selected")
        tiff.imsave(f"./saves/{name}.tiff", image)

    def shift_all(self):
        """
        Given all the translations, shift and save all the images.
        The images are saved in a folder called 'shifted_images', in the parent directory
            of the directory you submitted.
        E.g. if you say the data is in 'C:/Windows/Users/some_user/data',
            the shifted images will be in 'C:/Windows/Users/some_user/shifted_images'
        """
        parent_path: Path = Path(self.data_loc).parent
        shifted_images_path = parent_path.joinpath("shifted_images")
        for rnd in self.round_to_tiffs_dict:
            rnd_save_dir = shifted_images_path.joinpath(rnd)
            os.makedirs(rnd_save_dir, exist_ok=True)
            for image in self.round_to_tiffs_dict[rnd]:
                image_array = image.read_image()
                shifted_image = self.shift_image(image_array, rnd)
                save_path = os.path.join(rnd_save_dir, f"shifted_{image.key}")
                tiff.imwrite(save_path, data=shifted_image)
