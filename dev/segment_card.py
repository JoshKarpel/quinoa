import logging

logging.basicConfig()

from pathlib import Path
from pprint import pprint
import itertools

from tqdm import tqdm, trange

import numpy as np
import matplotlib.pyplot as plt
import skimage
import cv2 as cv

import quinoa as q

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

HERE = Path(__file__).parent.absolute()
DATA = HERE.parent / "data"
OUT = HERE / "out" / Path(__file__).stem


def process(path):
    logger.debug(f"Processing image at {path}")

    img_bgr = q.read_image(path)
    img_lab = q.convert_colorspace(img_bgr, cv.COLOR_BGR2LAB)
    img_b = img_lab[..., 2]  # just the blue-yellow channel

    thresh, thresholded = cv.threshold(cv.bitwise_not(img_b), 0, maxval = 255, type = cv.THRESH_OTSU)

    closed = cv.morphologyEx(
        thresholded,
        cv.MORPH_CLOSE,
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (50, 50))
    )

    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(closed, connectivity = 8)

    largest_label = max(range(1, num_labels), key = lambda label: stats[label, cv.CC_STAT_AREA])

    just_largest_component = np.where(labels == largest_label, closed, 0)

    edges = cv.Canny(cv.GaussianBlur(just_largest_component, (21, 21), 3), 10, 100, apertureSize = 7, L2gradient = True)

    lines = cv.HoughLinesP(
        edges,
        rho = 4,
        theta = np.deg2rad(1),
        threshold = 100,
        minLineLength = 200,
        maxLineGap = 500,
    ).squeeze()

    show_lines = img_bgr.copy()

    for idx, ((xs, ys, xe, ye), color) in enumerate(zip(lines[:4], itertools.repeat(q.PINK))):
        show_lines = q.draw_arrow(
            show_lines,
            (xs, ys),
            (xe, ye),
            color = color,
            thickness = 3,
        )
        show_lines = q.draw_text(
            show_lines,
            (xs - 20, ys - 20),
            str(idx),
            color = color,
            size = 5,
            thickness = 3,
        )

    q.write_image(show_lines, OUT / f"{path.stem}_card.jpg")


if __name__ == '__main__':
    image_paths = [path for path in DATA.iterdir() if path.suffix.lower() == '.jpg']

    for path in image_paths:
        process(path)
