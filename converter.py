import os
import numpy as np
import tifffile as tiff
import scipy.signal as signal
import scipy.ndimage as ndimage
from typing import List, Tuple, Dict

class NoneSelected(Exception):
    pass

class Converter:
    def __init__(self) -> None:
        self.data_loc: str = './data'
        self.filenames: List[str] = ["Alexa 488", "Alexa 647", "HOECHST 33342"] 
        self.dirs: List[str] = os.listdir(self.data_loc)
        self.all_tiffs: Dict[str, List] = {} # Mapping of the round to the image stacks
        self.hoechst: List = []
        self.translations: Dict[str, Tuple] = {self.dirs[0]: (0, 0)} # dict of rounds to translations
        self._stack_images()
        self._find_translations()
    
    def _stack_images(self) -> None:
        for i, dir in enumerate(self.dirs):
            for name in self.filenames:
                location = os.path.join(self.data_loc, dir)

                tiffs = os.listdir(location)
                images_names = [i for i in tiffs if name in i]
                images = np.array([np.array(tiff.imread(os.path.join(location, img))) for img in images_names])
                if images.shape[0] == 0:
                    continue
                res = np.max(images, axis=0)
                save_path = "./saves/" + f"{name}_{i}.tiff"

                if dir not in self.all_tiffs:
                    self.all_tiffs[dir] = []
                self.all_tiffs[dir].append((name, res))

                if name == "HOECHST 33342":
                    self.hoechst.append(res)

                tiff.imsave(save_path , res)
    
    def _cross_image(self, im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
        im1 = im1 - np.mean(im1)
        im2 = im2 - np.mean(im2)
        return signal.fftconvolve(im1, im2[::-1, ::-1], mode="same")
    
    def _find_translations(self):
        for i in range(1, len(self.hoechst)):
            corr = self._cross_image(self.hoechst[0], self.hoechst[i])
            x, y = np.unravel_index(np.argmax(corr), corr.shape)
            trans = (x - corr.shape[0] // 2, y - corr.shape[1] // 2)
            self.translations[self.dirs[i]] = trans
        self.hoechst = [] # clearing them out of memory to save space 
    
    def overlay(self, images: Dict) -> np.ndarray:
        """ Overlay the images """
        stacked = []
        for key in images:
            for image in images[key]:
                image = image.copy()
                if key != self.dirs[0]: # If not in the first round, then w need to shift the image
                    print("Shifting: ", key, "By: ", self.translations[key])
                    print(self.translations)
                    image = ndimage.shift(image, self.translations[key])
                stacked.append(image)
        if len(stacked) == 0:
            raise NoneSelected("No images selected")

        stack = np.stack(stacked, axis=0)
        return np.max(stack, axis=0)

    def save(self, image: np.ndarray, name: str) -> None:
        """ Save the image """
        if image is None:
            raise NoneSelected("No images selected")
        tiff.imsave(f"./saves/{name}.tiff", image)
                
