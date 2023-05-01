import os
from typing import List, Dict, Tuple

from image import Image, ImageKey
from collections import defaultdict


class NoneSelected(Exception):
    pass


def parse_all_images_in_folder(folder_path: str) -> List[Image]:
    """
    Parse all the images in a folder
    :rtype: object
    """
    images: List[Image] = []
    for image_name in os.listdir(folder_path):
        image = Image(image_name, folder_path)
        images.append(image)
    return images


def group_all_images_in_round(images: List[Image]) -> Dict[ImageKey, List[Image]]:
    """
    Group all the images in a round by their row, col, fov and channel
    :param images: List of images to parse
    :return: Dictionary of grouped images
    """
    groups: Dict[ImageKey, List[Image]] = defaultdict(
        list)  # Images that have the same row, col, fov and channel
    for image in images:
        key = image.key
        groups[key].append(image)
    return groups


def get_stacked_name(key: Tuple[int, int, int, int]) -> str:
    """
    Get the name of the stacked image
    :param key: The key of the image
    :return: The name of the stacked image
    """
    return f"r{key[0]}c{key[1]}f{key[2]}ch{key[3]}.tiff"


def is_hoesht(key: ImageKey):
    return key == ImageKey(0, 0, 0, 0)
