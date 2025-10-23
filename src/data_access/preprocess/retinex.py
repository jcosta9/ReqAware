"""Code adapted from https://github.com/RanaMostafaAbdElMohsen/Traffic_Sign_Recognition"""

from PIL import Image
import numpy as np
from .utils import PIL2OpenCV, OpenCV2PIL, check_pil_opencv


class Retinex(object):
    def __init__(self):
        pass

    def __call__(self, img):
        if check_pil_opencv(img) != "OpenCV":
            img = PIL2OpenCV(img)

        img = self.to_pil(self.apply_retinex((img)))
        open_cv_image = np.array(img)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        return OpenCV2PIL(open_cv_image)

    def apply_retinex(self, nimg):
        nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
        mu_g = nimg[1].max()
        nimg[0] = np.minimum(nimg[0] * (mu_g / float(nimg[0].max())), 255)
        nimg[2] = np.minimum(nimg[2] * (mu_g / float(nimg[2].max())), 255)
        return nimg.transpose(1, 2, 0).astype(np.uint8)

    def to_pil(self, nimg):
        return Image.fromarray(np.uint8(nimg))
