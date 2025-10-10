"""Code adapted from https://github.com/RanaMostafaAbdElMohsen/Traffic_Sign_Recognition"""

import skimage
from PIL import Image
import numpy as np

class Convert2Grayscale(object):
    """Converts an image to grayscale."""

    def __init__(self):
        pass

    def __call__(self, img):
        np_arr = skimage.color.rgb2gray(img)
        
        normalized_np_arr = (np_arr - np.min(np_arr)) / (np.max(np_arr) - np.min(np_arr))
        uint8_np_arr = (normalized_np_arr * 255).astype(np.uint8)
        img = Image.fromarray(uint8_np_arr)

        return img