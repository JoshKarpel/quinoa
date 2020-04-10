from pathlib import Path

import numpy as np
import cv2 as cv


def read_image(path: Path):
    return cv.imread(str(path))


def write_image(image: np.array, path: Path):
    return cv.imwrite(str(path), image)
