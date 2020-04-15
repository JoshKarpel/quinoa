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
    img_bgr = q.read_image(path)
    img_lab = q.convert_colorspace(img_bgr, cv.COLOR_BGR2LAB)

    corners = q.find_card_corners(image_bgr=img_bgr)
    new_corners = q.determine_new_corners(corners)
    crop_slice = q.corners_to_slice(new_corners)
    rectifier = q.get_rectifier(corners, new_corners)

    img_bgr_cropped = rectifier(img_bgr)[crop_slice]
    img_lab_cropped = rectifier(img_lab)[crop_slice]

    img_b_cropped = img_lab_cropped[..., 2]

    thresh, img_b_thresholded = cv.threshold(
        img_b_cropped, 0, maxval=255, type=cv.THRESH_OTSU
    )
    not_shadows = q.find_not_shadows(img_lab_cropped)

    cut_shadows = cv.bitwise_and(img_b_thresholded, not_shadows)

    num_labels, img_b_labels, stats, centroids = cv.connectedComponentsWithStats(
        cut_shadows, connectivity=8
    )

    show_labels = np.zeros_like(img_bgr_cropped)

    # use this to cut everything near the edges of the image
    boundary_width = 3
    edges = np.zeros_like(img_b_thresholded)
    edges[0:boundary_width, :] = 1
    edges[:, 0:boundary_width] = 1
    edges[-boundary_width:-1, :] = 1
    edges[:, -boundary_width:-1] = 1

    colors = [q.MAGENTA]
    for label, color in zip(range(1, num_labels), itertools.cycle(colors)):
        area = stats[label, cv.CC_STAT_AREA]
        if area < 40:
            continue

        in_label = np.where(label == img_b_labels, 1, 0).astype(np.uint8)

        if np.any(cv.bitwise_and(edges, in_label)):
            continue

        show_labels[label == img_b_labels] = color

    q.write_image(show_labels, OUT / f"{path.stem}_2.jpg")

    alpha = 2 / 3
    overlay = cv.addWeighted(img_bgr_cropped, alpha, show_labels, 1 - alpha, 0)
    q.write_image(overlay, OUT / f"{path.stem}_1.jpg")

    q.write_image(img_bgr_cropped, OUT / f"{path.stem}_0.jpg")



if __name__ == "__main__":
    image_paths = [path for path in DATA.iterdir() if path.suffix.lower() == ".jpg"]

    for path in image_paths:
        process(path)
