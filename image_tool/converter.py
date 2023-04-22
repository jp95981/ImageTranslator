import os
import numpy as np
import numpy.typing as npt
import tifffile as tiff
import scipy.signal as signal
from collections import defaultdict
from helper import ImageName, NoneSelected
from typing import Generator, List, Tuple, Dict, DefaultDict, Any, Set
import cv2 as cv  # type: ignore
from pathlib import Path


class Converter:
    def __init__(self, data_loc: str = "./data") -> None:
        self.data_loc: Path = Path(data_loc)
        self.round_directories: List[
            str
        ] = []  # A list of all of the paths for each round
        self.all_tiffs: DefaultDict[str, List[ImageName]] = defaultdict(
            list
        )  # Mapping of the round to the image stacks
        self.shifters: List[Any] = list()
        self.translations: Dict[str, Tuple[Any, Any]]

        self.filenames: List[str]
        self.proccess_data_location: Path
        self._image_setup()

    def _image_setup(self):
        self._set_round_paths()
        self._set_filenames()
        self._set_new_data_location()
        self._stack_images()
        self._find_translations()

    def _set_round_paths(self):
        self.round_directories = [
            x for x in os.listdir(self.data_loc) if not x.startswith(".")
        ]  # Remove any temp folders

        self.translations = {self.round_directories[0]: (0, 0)}

    def _set_filenames(self):
        suffixes = ("z1", "z2")
        folder_path = os.path.join(self.data_loc, self.round_directories[0])
        all_file_names = os.listdir(folder_path)
        all_file_names_no_ext = [os.path.splitext(x)[0] for x in all_file_names]

        non_duplicates = set()
        for filename in all_file_names_no_ext:
            if any(filename[-2:] == suffix for suffix in suffixes):
                non_duplicates.add(filename[:-2])
            else:
                non_duplicates.add(filename)
        self.filenames = list(non_duplicates)

    def _set_new_data_location(self):
        data_parent_path = Path(self.data_loc).parent.absolute()
        data_new_path = Path.joinpath(data_parent_path, "processed_images")
        data_new_path.mkdir(exist_ok=True)
        self.proccess_data_location = data_new_path

    def _get_image_cluster(
        self, image_names: List[str], location: Path, chunk_size: int = 200
    ) -> Generator[npt.NDArray, None, None]:
        for i in range(0, len(image_names), chunk_size):
            chunked_name = image_names[i : i + chunk_size]
            yield np.array(
                [
                    np.array(tiff.imread(os.path.join(location, img)))
                    for img in chunked_name
                ]
            )

    def _stack_images(self) -> None:
        """
        Stacks images of different field of views together
        For each round find all the images of the same well and stack them together
        """

        for i, round_directory_name in enumerate(self.round_directories):
            round_directory = Path.joinpath(
                self.proccess_data_location, round_directory_name
            )
            round_directory.mkdir(exist_ok=True)
            for name in self.filenames:
                raw_image_location: Path = self.data_loc.joinpath(round_directory_name)
                all_tiffs_in_folder = os.listdir(raw_image_location)
                image_names = [tiff_ for tiff_ in all_tiffs_in_folder if name in tiff_]

                for images in self._get_image_cluster(image_names, raw_image_location):
                    # Any empty images
                    if images.shape[0] == 0:
                        continue

                    # Stack the images
                    res: npt.NDArray[np.float64] = np.max(images, axis=0)
                    tiff_name: ImageName = ImageName(f"{name}_{i}.tiff")
                    self.all_tiffs[round_directory_name].append(tiff_name)
                    if name == "HOECHST 33342":
                        self.shifters.append(res)
                    save_directory = round_directory.joinpath(tiff_name)
                    tiff.imwrite(save_directory, res)

    def _cross_image(self, im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
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
        So translate rounds 2 onwards so that they line up with round 1 images
        """
        for i in range(1, len(self.shifters)):
            # Find correlation between current image, and the one from rnd 1
            corr = self._cross_image(self.shifters[0], self.shifters[i])
            # the brightest point of the convoled image is the new image center
            x, y = np.unravel_index(np.argmax(corr), corr.shape)
            # So the difference between the current image center and brightest point
            # is the translation
            trans: Tuple[Any, Any] = (x - corr.shape[0] // 2, y - corr.shape[1] // 2)
            curr_rnd = self.round_directories[i]
            self.translations[curr_rnd] = trans
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

    def overlay(self, images: Dict[str, List[ImageName]]) -> np.ndarray:
        """
        Find the selected images and overlay them (i.e. the maximum pixels)

        Args:
            images (Dict[str, List[ImageName]]): Dictionary of all the images RoundName -> [ImagesNames]

        Raises:
            NoneSelected: If no images are selected

        Returns:
            np.ndarray: The overlayed image of all those selected
        """
        stacked = []
        for rnd in images:
            for image in images[rnd]:
                img_path = os.path.join(self.proccess_data_location, rnd, image)
                image = tiff.imread(img_path)
                if (
                    rnd != self.round_directories[0]
                ):  # If not in the first round, then w need to shift the image
                    image = self.shift_image(image, rnd)
                stacked.append(image)
        if len(stacked) == 0:
            raise NoneSelected("No images selected")

        stack = np.stack(stacked, axis=0)
        max_stack = np.max(stack, axis=0)
        return max_stack

    def save(self, image: np.ndarray, name: str) -> None:
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
        Given all the translations, shift and save all of the images
        The images are saved in a folder called 'shifted_images', in the parent directory
            of the directory you submitted.
        e.g. if you entered the data is in 'C:/Windows/Users/some_user/data',
            the shifted images will be in 'C:/Windows/Users/some_user/shifted_images'
        """
        parent_path: Path = Path(self.data_loc).parent.absolute()
        shifted_images_path = parent_path.joinpath("shifted_images")
        for rnd in self.all_tiffs:
            rnd_save_dir = shifted_images_path.joinpath(rnd)
            os.makedirs(rnd_save_dir, exist_ok=True)
            for image_name in self.all_tiffs[rnd]:
                image_path = self.proccess_data_location.joinpath(rnd).joinpath(
                    image_name
                )
                image = tiff.imread(image_path)
                shifted_image = self.shift_image(image, rnd)
                save_path = os.path.join(rnd_save_dir, f"shifted_{image_name}")
                tiff.imwrite(save_path, data=shifted_image)
