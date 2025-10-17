import cv2
from PIL import Image
import numpy as np


def check_pil_opencv(img):
    if type(img) == np.ndarray:
        return "OpenCV"
    if isinstance(img, Image.Image):
        return "PIL"
    return "unkown"


def PIL2OpenCV(pil_img):
    img = pil_img.convert("RGB")
    open_cv_image = np.array(img)
    return open_cv_image[:, :, ::-1].copy()


def OpenCV2PIL(ocv_img):
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(ocv_img, cv2.COLOR_BGR2RGB)

    # Create a PIL image from the RGB array
    return Image.fromarray(rgb_image)
