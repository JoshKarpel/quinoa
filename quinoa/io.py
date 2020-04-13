import logging
from pathlib import Path

import numpy as np
import cv2 as cv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def read_image(path: Path):
    img = cv.imread(str(path))
    logger.debug(f"Loaded image from {path}")
    return img


def write_image(image: np.array, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rv = cv.imwrite(str(path), image)
    logger.debug(f"Wrote image (size {image.shape}) to {path}")
    return rv
