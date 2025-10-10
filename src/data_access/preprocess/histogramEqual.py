"""Code adapted from https://github.com/RanaMostafaAbdElMohsen/Traffic_Sign_Recognition"""

import cv2
from .utils import PIL2OpenCV, OpenCV2PIL, check_pil_opencv

class HistogramEqualization(object):
    def __init__(self):
        pass

    def __call__(self, img):
        if check_pil_opencv(img) != "OpenCV":
            img = PIL2OpenCV(img)

        img = self.histogram_equalization(img)

        return OpenCV2PIL(img)

    def histogram_equalization(self, img):
        img_to_yuv = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
        img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
        return cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2RGB)