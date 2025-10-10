"""Code adapted from https://github.com/RanaMostafaAbdElMohsen/Traffic_Sign_Recognition"""

import numpy as np
import cv2
from .utils import PIL2OpenCV, OpenCV2PIL, check_pil_opencv

class EdgeEnhancement(object):
    def __init__(self):
        pass

    def __call__(self, img):
        if check_pil_opencv(img) != "OpenCV":
            img = PIL2OpenCV(img)

        img = self.edge_enhancement(img)

        return OpenCV2PIL(img)
    
    def edge_enhancement(self, img):
        kernel = np.array([[-1,-1,-1,-1,-1],
                                [-1,2,2,2,-1],
                                [-1,2,8,2,-1],
                                [-2,2,2,2,-1],
                                [-1,-1,-1,-1,-1]])/8.0
        img = cv2.filter2D(img, -1, kernel)
        image=cv2.filter2D(img, -1, kernel)
        return image