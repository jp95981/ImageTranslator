import os
import numpy as np
import tifffile as tiff
import scipy.signal as signal
import scipy.ndimage as ndimage
from typing import List, Tuple, Dict
import cv2 as cv
import time
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
        start = time.time()
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
        print(f"Time taken to stack images: {time.time() - start}")
    
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
    
    def shift_image(self, im1: np.ndarray, rnd: str) -> np.ndarray:
        start = time.time()
        print(self.translations[rnd])
        x, y = self.translations[rnd]
        transform = np.float32([[1,0, -x], [0,1, -y]])
        shiffted = cv.warpAffine(im1, transform, (im1.shape[0], im1.shape[1]))
        print(f"Time taken to shift image: {time.time() - start}")
        return shiffted

    def overlay(self, images: Dict) -> np.ndarray:
        """ Overlay the images """
        start = time.time()
        stacked = []
        for rnd in images:
            for image in images[rnd]:
                image = image.copy()
                if rnd != self.dirs[0]: # If not in the first round, then w need to shift the image
                    image = self.shift_image(image, rnd)
                stacked.append(image)
        if len(stacked) == 0:
            raise NoneSelected("No images selected")

        stack = np.stack(stacked, axis=0)
        max_stack =  np.max(stack, axis=0)
        print(f"Time taken to overlay images: {time.time() - start}")
        return max_stack

    def save(self, image: np.ndarray, name: str) -> None:
        """ Save the image """
        if image is None:
            raise NoneSelected("No images selected")
        tiff.imsave(f"./saves/{name}.tiff", image)
                
